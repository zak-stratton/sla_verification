#!/usr/local/bin/python3

import logging
# Used for command-line argument passing.
import argparse
# Used for time.time()
import time
# Used for datetime.datetime.now()
import datetime
# Used to introduce randomness to the number of requests issued by the program via the
# --numRequestsUncertainty parameter.
import random
# Used to bind additional arguments to a callback function.
import functools
# Used to enable clean Ctrl+C bring-down of the program.
import signal
# Library used to launch and manage concurrent service requests.
from concurrent import futures
# Event used to wait until entirety of an multi-part response stream is received.
from multiprocessing import Event
# Used to synchronize access of multiple threads to a single object.
from threading import Lock
# Used to compute percentiles of a list of values.
import numpy as np
# Used for managing HTTP requests & responses.
import requests


# -------------------------------------------------------------------------------------------------
# RequestFactory, a read-only class, will take in a (svcName, appName) input tuple and return to the
# caller a (url, params) output tuple.  All request-specific logic is encapsulated in this class.

APP_A = "APP-A"
APP_B = "APP-B"
APP_C = "APP-C"
SVC_1 = "SVC-1"
SVC_2 = "SVC-2"
SINGLE_RESPONSE_SVCS = [SVC_1]
MULTI_RESPONSE_SVCS = [SVC_2]
APPLICATION_AND_SERVICE = [
    APP_A + '_' + SVC_1,
    APP_B + '_' + SVC_1,
    APP_C + '_' + SVC_2
]


class RequestFactory:
    def __init__(self):
        self.svcAndAppToUserID = {
            SVC_1: {
                APP_A: "genericUserId_withPermissionsSetM",
                APP_B: "genericUserId_withPermissionsSetN",
            },
            SVC_2: {
                APP_C: "genericUserId_withPermissionsSetP"
            }
        }
        self.svcAndAppToAppID = {
            SVC_1: {
                APP_A: "appAId",
                APP_B: "appBId"
            },
            SVC_2: {
                APP_C: "appCId"
            }
        }

    # At the end of this method, we will be returning a (url, params) output tuple.  If the set
    # of services were not HTTP services, this function could be re-worked.
    def getUrlAndParams(self, svcName, appName):
        if svcName == SVC_1:
            # Do any necessary setup for SVC_1, pulling in params from other services, etc.  This
            # setup may be dependent on the app.
            # No additional setup is specified here.
            url = "https://httpbin.org/get"
            params = {
                "userID": self.svcAndAppToUserID[svcName][appName],
                "appID": self.svcAndAppToAppID[svcName][appName],
                "paramForService": "dummy",
                "anotherParamForService": "dummy",
                "yetAnotherParamForService": "dummy"
            }
        elif svcName == SVC_2:
            # Do any necessary setup for SVC_2, pulling in params from other services, etc.  This
            # setup may be dependent on the app.
            # No additional setup is specified here.
            url = "https://httpbin.org/get"
            params = {
                "userID": self.svcAndAppToUserID[svcName][appName],
                "appID": self.svcAndAppToAppID[svcName][appName],
                "paramForService": "dummy"
            }
        else:
            errorStr = "ALARM=PROGRAMMER_ERROR_UNRECOGNIZED_ID.  RequestFactory." + \
                       "getUrlAndParams(svcName, appName) called with unrecognized IDs " + \
                       f"svcName={svcName}, appName={appName}."
            logger.error(errorStr)
            raise Exception(errorStr)
        return url, params


# We instantiate a single RequestFactory object, available in global space.  This is because the
# RequestFactory will need to be accessible (read-only) by the threads in the ThreadPool.
g_RequestFactory = RequestFactory()


# -------------------------------------------------------------------------------------------------
# ErrorResponseTracker, thread-safe, used to track error responses received in response to
# launched service requests.

class ErrorResponseTracker:
    def __init__(self):
        self.count = 0
        self.mutex = Lock()
        self.errorResponseThreshold = 10

    def increment(self):
        with self.mutex:
            self.count += 1
            return self.count

    def exitProgram(self):
        with self.mutex:
            return self.count >= self.errorResponseThreshold

    # Validity of the response is tested by examining the HTTP status code.  If the set of services
    # were not HTTP services, this function could be re-worked.
    def isErrorResponse(self, svcName, response):
        if svcName == SVC_1:
            return response.status_code != requests.codes.ok
        elif svcName == SVC_2:
            return response.status_code != requests.codes.ok
        else:
            errorStr = "ALARM=PROGRAMMER_ERROR_UNRECOGNIZED_ID.  ErrorResponseTracker." + \
                       "isErrorResponse(svcName, response) called with unrecognized " + \
                       f"svcName={svcName}."
            logger.error(errorStr)
            raise Exception(errorStr)


# We instantiate a single ErrorResponseTracker object, available in global space.  This is because
# all service response handling will need to access the object, and each response is handled in a
# unique thread.
g_ErrorResponseTracker = ErrorResponseTracker()


# -------------------------------------------------------------------------------------------------
# LatencyTracker, thread-safe, for use in *RECORDING* the latency between:
#   1) Sending a service request and receipt of first response.
#   2) Any two consecutive responses in a stream of multi-part responses.
# The LatencyTracker provides the ability to output a percentile (or list of percentiles), measured
# on the distribution of the data it's tracking.

class LatencyTracker:
    def __init__(self):
        self.initialRespLatencyValues = np.array([])
        self.streamRespLatencyValues = np.array([])
        self.mutex = Lock()
        self.appInitialRespLatencyThresholds = {
            # Threshold for 90th percentile latency, in seconds, of initial received response.
            APP_A: 7,
            APP_B: 5,
            APP_C: 4
        }
        self.appStreamRespLatencyThresholds = {
            # Threshold for 90th percentile latency, in seconds, between two consecutive
            # responses in a stream of multi-part responses.
            APP_C: 2
        }

    def addValue(self, value, forInitialResponse):
        with self.mutex:
            if forInitialResponse:
                self.initialRespLatencyValues = np.append(self.initialRespLatencyValues, value)
            else:
                self.streamRespLatencyValues = np.append(self.streamRespLatencyValues, value)

    def getPercentiles(self, percs, forInitialResponse):
        # The np.percentile function takes either a single percentile, or a list of percentiles.
        # We have this getPercentiles method follow the same input approach.
        with self.mutex:
            if forInitialResponse:
                latencyValues = self.initialRespLatencyValues
            else:
                latencyValues = self.streamRespLatencyValues
            if latencyValues.size == 0:
                return [None for perc in percs] if isinstance(percs, list) else None
            values = np.percentile(latencyValues, percs)
            return values.tolist() if isinstance(percs, list) else values

    def exitProgram(self, appName):
        # Mutex locking is done in the 'getPercentiles(...)' call.
        initialRespLatency_90thPerc = self.getPercentiles(90, forInitialResponse=True)
        if initialRespLatency_90thPerc is not None and \
           initialRespLatency_90thPerc > self.appInitialRespLatencyThresholds[appName]:
            return True
        streamRespLatency_90thPerc = self.getPercentiles(90, forInitialResponse=False)
        if streamRespLatency_90thPerc is not None and \
           streamRespLatency_90thPerc > self.appStreamRespLatencyThresholds[appName]:
            return True
        return False


# We instantiate a single LatencyTracker object, available in global space.  This is because all
# service response handling will need to access the object, and each response is handled in a
# unique thread.
g_LatencyTracker = LatencyTracker()


# -------------------------------------------------------------------------------------------------
# Stopwatch, thread-safe, for use in *TIMING* the latency between:
#   1) A service request and the first response,
#   2) Any two consecutive responses in a stream of multi-part responses
# The Stopwatch provides the ability to restart the timer, simultaneously outputting the duration,
# in seconds, of the time interval just timed.

class Stopwatch:
    def __init__(self):
        self.startTime = time.time()
        self.initialResponseHasBeenClocked = False
        self.mutex = Lock()

    def restart(self):
        with self.mutex:
            currentTime = time.time()
            timePassed = currentTime - self.startTime
            self.startTime = currentTime
            isInitialResponse = False
            if not self.initialResponseHasBeenClocked:
                isInitialResponse = True
                self.initialResponseHasBeenClocked = True
            return timePassed, isInitialResponse


# -------------------------------------------------------------------------------------------------
# Functionality for sending requests / receiving responses.
#  --The 'sendRequest_singleResp' and 'sendRequest_multipartResp' functions are executed by a
#    ThreadPool.  Each of those functions results in a future object.  If an exception is raised
#    within either function, the future will capture it.  If no exception is raised, the future
#    will capture any return arguments from those functions.
#  --The 'sendRequest_singleResp' future has the 'examineSingleResponse' callback attached to it.
#    When the future completes, 'examineSingleResponse' runs.
#  --The 'sendRequest_multipartResp' future has the 'testFutureForException' callback attached to
#    it.  This will guard against any exceptions raised within 'sendRequest_multipartResp' prior
#    to the service call.
#    The service call attaches the 'examineMultipartResponse' callback to the multi-part response
#    stream.  'examineMultipartResponse' runs each time a response is received.

# NOTE this function is attached as a callback via the concurrent.futures 'add_done_callback'
# functionality.  Per the concurrent.futures documentation, any exceptions surfaced within this
# function will be ignored.
def testFutureForException(future):
    # If the future yields an exception, proxy it as an error response.
    if future.exception():
        global g_ErrorResponseTracker
        cumErrorCount = g_ErrorResponseTracker.increment()
        logger.warn("Cumulative error response count : permitted threshold = "
                    f"{cumErrorCount} : {g_ErrorResponseTracker.errorResponseThreshold}  |  "
                    "sendRequest_multipartResp function resulted in exception prior to "
                    f"the service request, exception={future.exception()}")


# NOTE this function is attached as a callback via the concurrent.futures 'add_done_callback'
# functionality.  Per the concurrent.futures documentation, any exceptions surfaced within this
# function will be ignored.
def examineSingleResponse(svcName, responseFuture):
    # If the future yields an exception, proxy it as an error response, and return.  Otherwise,
    # access the response and the stopwatch from the future.
    global g_ErrorResponseTracker
    if responseFuture.exception():
        cumErrorCount = g_ErrorResponseTracker.increment()
        logger.warn("Cumulative error response count : permitted threshold = "
                    f"{cumErrorCount} : {g_ErrorResponseTracker.errorResponseThreshold}  |  "
                    f"sendRequest_singleResp resulted in exception={responseFuture.exception()}")
        return
    response, stopwatch = responseFuture.result()

    # Track the latency.  The stopwatch object will specify whether this is the first time it
    # has been restarted, and the g_LatencyTracker object will track the result accordingly.
    global g_LatencyTracker
    timePassed, isInitialResponse = stopwatch.restart()
    g_LatencyTracker.addValue(timePassed, isInitialResponse)

    # If the response indicates an error, track it.
    if g_ErrorResponseTracker.isErrorResponse(svcName, response):
        cumErrorCount = g_ErrorResponseTracker.increment()
        logger.warn("Cumulative error response count : permitted threshold = "
                    f"{cumErrorCount} : {g_ErrorResponseTracker.errorResponseThreshold}  |  "
                    f"sendRequest_singleResp resulted in errorResponse={response}")


def examineMultipartResponse(svcName, stopwatch, event, response, exception, finalFlag):
    # If an exception occurs, proxy it as an error response, and return.  Need to also check whether
    # the exception is the final item in the multi-part response stream.
    global g_ErrorResponseTracker
    if exception:
        cumErrorCount = g_ErrorResponseTracker.increment()
        logger.warn("Cumulative error response count : permitted threshold = "
                    f"{cumErrorCount} : {g_ErrorResponseTracker.errorResponseThreshold}  |  "
                    f"sendRequest_multipartResp resulted in exception={exception}")
        if finalFlag:
            event.set()
        return

    # Track the latency.  The stopwatch object will specify whether this is the first time it
    # has been restarted, and the g_LatencyTracker object will track the result accordingly.
    global g_LatencyTracker
    timePassed, isInitialResponse = stopwatch.restart()
    g_LatencyTracker.addValue(timePassed, isInitialResponse)

    # If the response indicates an error, track it.
    if g_ErrorResponseTracker.isErrorResponse(svcName, response):
        cumErrorCount = g_ErrorResponseTracker.increment()
        logger.warn("Cumulative error response count : permitted threshold = "
                    f"{cumErrorCount} : {g_ErrorResponseTracker.errorResponseThreshold}  |  "
                    f"sendRequest_multipartResp resulted in errorResponse={response}")

    # The event object is used to keep the service request context alive until the entirety of
    # the multi-part response stream is received.  Thus, when the final response is received, we
    # mark the event, indicating the request context can now be brought down.
    if finalFlag:
        event.set()


def sendRequest_singleResp(svcName, appName):
    url, params = g_RequestFactory.getUrlAndParams(svcName, appName)
    stopwatch = Stopwatch()
    response = requests.get(url=url, params=params)
    return response, stopwatch


def sendRequest_multipartResp(svcName, appName):
    url, params = g_RequestFactory.getUrlAndParams(svcName, appName)
    stopwatch = Stopwatch()
    # This Event() object is needed to keep the service request context alive until the multi-part
    # response stream has been received in its entirety.  Thus, the event must be passed into the
    # callback, which will mark it when the the multi-part response stream has completed.
    event = Event()
    # We bind (svcName, stopwatch, event) to the 'examineMultipartResponse' function, which will
    # serve as our callback when each response in the multi-part stream is received.
    # NOTE right now, the service called is dummied out with a simple single response service.
    # For a multi-part response service, this would be re-worked, and the callback would be used to
    # handle each arriving response in the stream.
    callback = functools.partial(examineMultipartResponse, svcName, stopwatch, event)
    requests.get(url=url, params=params)
    event.set()
    event.wait()


# -------------------------------------------------------------------------------------------------
# Primary program functionality.

# This function examines the 'overrides'--which were specified to the program via the command
# line--to determine whether the present time falls within an override interval.  If it does, the
# function returns the number of requests specified in the override.
def testForOverride(overrides):
    overrideApplicable = False
    overrideNumReqs = None
    if overrides:
        currTime = datetime.datetime.now()
        for override in overrides:
            o = override[0]
            startTime = currTime.replace(hour=o["startHour"],
                                         minute=o["startMinute"],
                                         second=0)
            endTime = currTime.replace(hour=o["endHour"],
                                       minute=o["endMinute"],
                                       second=0)
            if startTime < currTime and currTime < endTime:
                overrideApplicable = True
                overrideNumReqs = o["numRequests"]
                break
    return overrideApplicable, overrideNumReqs


def main(args):
    appName = args.application_service[0].split('_')[0]
    svcName = args.application_service[0].split('_')[1]
    timeInterval = args.timeInterval[0]
    nominalNumReqs = args.numRequests[0]
    uncertainty = args.numRequestsUncertainty[0] if args.numRequestsUncertainty else None
    overrides = args.override if args.override else None
    logger.info(f"Initiating run with: application='{appName}', service='{svcName}', "
                f"timeInterval={timeInterval}sec, numRequests={nominalNumReqs}, "
                f"numRequestsUncertainty={uncertainty}, overrides={overrides}.")

    # This ThreadPool is used to launch service requests.  The total outstanding requests at any one
    # time is thus limited to the number of workers in the ThreadPool.  This serves as a check on
    # user inputs to the program: if such inputs would cause excessive loading of the tested
    # service, max_workers prevents it.
    with futures.ThreadPoolExecutor(max_workers=50) as executor:
        global g_ErrorResponseTracker, g_LatencyTracker
        # This while loop is entered every 'timeInterval' seconds.  The code block within it is
        # responsible for sending a number of service requests, as arrived at via the combination of
        # inputs to the program.
        while True:
            # Compute the time between sending service requests over the 'timeInterval'.
            overrideApplicable, overrideNumReqs = testForOverride(overrides)
            if overrideApplicable:
                numReqs = overrideNumReqs
            else:
                numReqs = nominalNumReqs
            if uncertainty:
                # The 'uncertainty' parameter is specified to the program as a 0.XX float, and
                # incidates a maximum percentage of the baseline number of requests, to then be
                # introduced on top of the baseline as an addition / subtraction.
                # So, we randomly choose a number between 0 and the 'uncertainty', and then
                # randomly choose whether to add it to / subtract it from the baseline.
                numReqsUncertaintyAddition = round(numReqs * random.uniform(0, uncertainty))
                if random.randint(0, 1):
                    numReqsUncertaintyAddition *= -1
                numReqs += numReqsUncertaintyAddition
            timeBetweenReqs = timeInterval / float(numReqs)

            # Send requests, via the ThreadPool, waiting the requisite time in between.
            startTimeStr = datetime.datetime.now().strftime('%H:%M:%S')
            countReqs = 0
            endTime = time.time() + timeInterval
            while time.time() < endTime:
                countReqs += 1
                if svcName in SINGLE_RESPONSE_SVCS:
                    # We bind 'svcName' to the 'examineSingleResponse' function, which serves as our
                    # callback when the service response is received.
                    # Any exceptions raised within 'sendRequest_singleResp' must be handled by callback.
                    callback = functools.partial(examineSingleResponse, svcName)
                    executor.submit(sendRequest_singleResp, svcName, appName).add_done_callback(callback)
                elif svcName in MULTI_RESPONSE_SVCS:
                    # Any exceptions raised within 'sendRequest_multipartResp' must be handled by callback.
                    callback = testFutureForException
                    executor.submit(sendRequest_multipartResp, svcName, appName).add_done_callback(callback)
                time.sleep(timeBetweenReqs)
            endTimeStr = datetime.datetime.now().strftime('%H:%M:%S')
            initLatencyStats = g_LatencyTracker.getPercentiles([0, 50, 90, 100], forInitialResponse=True)
            streamLatencyStats = g_LatencyTracker.getPercentiles([0, 50, 90, 100], forInitialResponse=False)
            logger.info(f"Sent numRequests={countReqs} over timeInterval={startTimeStr}-{endTimeStr}.  "
                        f"Response Latency Distribution [min, 50th, 90th, max] --> "
                        f"Initial{[round(stat,2) for stat in initLatencyStats if stat]} | "
                        f"Stream{[round(stat,3) for stat in streamLatencyStats if stat]}.")
            # Before continuing to send requests during the next 'timeInterval', have we overloaded
            # the service, or failed our latency SLAs?  If so, terminate this program.
            if g_ErrorResponseTracker.exitProgram() or g_LatencyTracker.exitProgram(appName):
                logger.error(
                    "ALARM=PROGRAM_EXIT_ON_SERVICE_OVERLOADING. "
                    f"Error Count = {g_ErrorResponseTracker.count}, "
                    "Response Latency 90th Perc --> "
                    f"Initial[{round(initLatencyStats[2],2) if initLatencyStats[2] else None}] | "
                    f"Stream[{round(streamLatencyStats[2],3) if streamLatencyStats[2] else None}],"
                    " inferring service overload and TERMINATING."
                )
                break
    logger.error("***********Program TERMINATING***********")


if __name__ == "__main__":
    # Parse and validate the program inputs.
    def timeIntervalValidation(intNum):
        intNum = int(intNum)
        if intNum < 0 or intNum > 60:
            raise argparse.ArgumentTypeError("Time interval must be between 0-60 seconds.")
        return intNum

    def numRequestsValidation(intNum):
        intNum = int(intNum)
        if intNum < 0 or intNum > 1000:
            raise argparse.ArgumentTypeError("Baseline number of requests must be between 0-1000.")
        return intNum

    def numRequestsUncertaintyValidation(floatNum):
        floatNum = float(floatNum)
        if floatNum < 0 or floatNum > 1:
            raise argparse.ArgumentTypeError("Request uncertainty must be percentage between 0-1.")
        return floatNum

    def overrideValidation(senSequence):
        errorStr = f"Input S,E,N sequence {senSequence} must be of the form HH:MM,HH:MM," + \
                   "integer, where N the number of requests is between 0-1000."
        senSequence = senSequence.split(',')
        if len(senSequence) != 3:
            raise argparse.ArgumentTypeError(errorStr)
        try:
            startT = [int(num) for num in senSequence[0].split(':')]
            endT = [int(num) for num in senSequence[1].split(':')]
            numRequests = int(senSequence[2])
        except:
            raise argparse.ArgumentTypeError(errorStr)
        if len(startT) != 2 or startT[0] < 0 or startT[0] > 23 or startT[1] < 0 or startT[1] > 59:
            raise argparse.ArgumentTypeError(errorStr)
        if len(endT) != 2 or endT[0] < 0 or endT[0] > 23 or endT[1] < 0 or endT[1] > 59:
            raise argparse.ArgumentTypeError(errorStr)
        if numRequests < 0 or numRequests > 1000:
            raise argparse.ArgumentTypeError(errorStr)
        return {"startHour": startT[0], "startMinute": startT[1],
                "endHour": endT[0], "endMinute": endT[1],
                "numRequests": numRequests}

    parser = argparse.ArgumentParser(description="Program used to simulate load on a service.")
    parser.add_argument("--application_service", "-a_s",
                        required=True,
                        nargs=1,
                        choices=APPLICATION_AND_SERVICE)
    parser.add_argument("--timeInterval", "-ti",
                        required=True,
                        nargs=1,
                        type=timeIntervalValidation,
                        metavar="integer",
                        help="Time interval, in seconds, over which to send the specified number "
                             "of requests.")
    parser.add_argument("--numRequests", "-nr",
                        required=True,
                        nargs=1,
                        type=numRequestsValidation,
                        metavar="integer",
                        help="Baseline number of requests to send in the specified time interval.")
    parser.add_argument("--numRequestsUncertainty", "-nru",
                        required=False,
                        nargs=1,
                        type=numRequestsUncertaintyValidation,
                        metavar="0.XX",
                        help="A percentage of the baseline number of requests, uniformly chosen "
                             "between (0, numRequestsUncertainty), is added to / subtracted from "
                             "the baseline number of requests.")
    parser.add_argument("--override", "-o",
                        required=False,
                        nargs=1,
                        type=overrideValidation,
                        action='append',
                        metavar=("HH:MM, HH:MM, integer"),
                        help="The argument input is a S, E, N sequence.  Within the time period "
                             "indicated by (S,E), the timeInterval, which usually has numRequests "
                             "sent over it, will now have N requests sent over it instead.")
    args = parser.parse_args()

    # Setup log.  This log format is prepared for ingestion by Splunk, and easy identification in
    # act.log if an error statement is surfaced.
    logger = logging.getLogger()
    logFormat = ("%(asctime)s (%(thread)d) %(levelname)s %(name)s "
                 "%(filename)s:%(lineno)d %(message)s")
    formatter = logging.Formatter("traffic_simulator.py\t" + logFormat)
    streamHandler = logging.StreamHandler()
    fileHandler = logging.FileHandler(filename="./traffic_simulator.log")
    streamHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)
    streamHandler.setLevel(logging.WARN)
    fileHandler.setLevel(logging.INFO)
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)
    logger.setLevel(logging.INFO)

    # Set the signal handler for Ctrl+C to be the default.
    # This allows us to to kill the program using Ctrl+C, gracefully.
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # Run the program.
    main(args)
