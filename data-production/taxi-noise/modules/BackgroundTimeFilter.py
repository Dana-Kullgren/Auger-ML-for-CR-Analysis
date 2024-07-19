from icecube import icetray, dataclasses
from icecube.icetray.i3logging import log_info, log_warn, log_trace, log_fatal
from datetime import datetime

class BackgroundTimeFilter(icetray.I3Module):
    """
    Reads in the times at which background noise files were recorded and only keeps the first 12 hours
    of each day. The last two hours per day tend to be noisier than expected so they are rejected.
    This module should be used if background noise cannot be adequately reduced by bandpass filtering.
    """

    def __init__(self, ctx):
        icetray.I3Module.__init__(self, ctx)

        self.outputName = ""
        self.AddParameter("OutputName", "Name of the output antenna data map", self.outputName)

        self.taxiFile = ""
        self.AddParameter("TaxiFile", "I3 file(s) with TAXI waveforms", self.taxiFile)

        self.HourRange = ""
        self.AddParameter("HourRange", "Range of hours that TAXI files will excepted from. All files outside this time range are rejected.", self.HourRange)

        
    def Configure(self):
        log_info("Configuring " + self.name)
        self.outputName = self.GetParameter("OutputName")
        self.taxiFile = self.GetParameter("TaxiFile")
        self.HourRange = self.GetParameter("HourRange")

        # Load up the waveforms here
        # self.backgrounds = TAXIBackgroundReader(self.taxiFile)
        # self.backgrounds.LoadAllFiles()


    def DAQ(self, frame):
        
        log_trace("({0}) Filtering TAXI files by time".format(self.name))

        def isNowInTimePeriod(i3Time, startRange=self.HourRange[0], endRange=self.HourRange[1]): 
            if startRange < endRange: 
                return i3Time >= startRange and i3Time <= endRange 
            else: 
                #Over midnight: 
                return i3Time >= startRange or i3Time <= endRange 

        taxi_files = []

        for taxi_file in self.taxiFile:
            print(f'taxi_file = {taxi_file}')

            start_time = frame['I3EventHeader'].start_time
            print(f'start: {start_time}')
            start_time = datetime.strptime(str(start_time)[:-18], '%Y-%m-%d %H:%M:%S')

            end_time = frame['I3EventHeader'].end_time
            print(f'end: {end_time}')
            end_time = datetime.strptime(str(end_time)[:-18], '%Y-%m-%d %H:%M:%S')

            print(f'start_time = {start_time}')
            print(f'end_time = {end_time}')

            if isNowInTimePeriod(start_time.time()) and isNowInTimePeriod(end_time.time()):
                print("It's an old code but it checks out")
                taxi_files.append(taxi_file)
                # self.outputName = self.taxiFile

        self.outputName = taxi_files
        print(f'self.taxiFile = {self.taxiFile}')
        print(f'self.outputName = {self.outputName}')

        self.PushFrame(frame)
        log_trace("({0}) Finished filtering TAXI files by time".format(self.name))
