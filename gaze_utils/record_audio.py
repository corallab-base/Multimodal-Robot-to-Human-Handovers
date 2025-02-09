import os
from matplotlib import pyplot as plt
import pyaudio
import numpy as np
from scipy.io import wavfile


class Recorder():
    def __init__(self):
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.sample_rate = 44100
        self.chunk = int(0.03 * self.sample_rate)
        self.START_KEY = 's'
        self.STOP_KEY = 'q'

        try: os.remove('gaze_utils/audio.wav')
        except OSError: pass


    def record(self):
        
        recorded_data = []
        p = pyaudio.PyAudio()

        stream = p.open(format=self.audio_format, channels=self.channels,
                        rate=self.sample_rate, input=True,
                        frames_per_buffer=self.chunk)
        
        print('\n\nSpeak now!\n')

        speaking = False

        stoppped_speaking_duration = 0

        energies = []

        try:
            while True:
                data = stream.read(self.chunk)

                audio_chunk = np.frombuffer(data, dtype=np.int16)

                energy = np.sum(audio_chunk.astype(np.float32) ** 2) / len(audio_chunk)
                energies.append(energy)

                # print('energy', energy)
                
                # If energy exceeds the threshold, consider it as speaking
                if energy > 0.3e6:
                    speaking = True
                    stoppped_speaking_duration = 0

                recorded_data.append(data)

                # If speaking was detected and energy falls below the threshold, stop recording
                if energy < 1e5 and speaking:
                    stoppped_speaking_duration += 1
                if stoppped_speaking_duration > 60:
                    break

                # plt.plot(range(len(energies)), energies)
                # plt.pause(0.001)
                
        except KeyboardInterrupt:
            print('Not implemented')
            pass

        finally:
            print("Speech Recorded!")
            # stop and close the stream
            stream.stop_stream()
            stream.close()
            p.terminate()
            # Convert recorded data to numpy array
            recorded_data = [np.frombuffer(frame, dtype=np.int16) for frame in recorded_data]
            wav = np.concatenate(recorded_data, axis=0)
            wavfile.write('gaze_utils/audio.wav', self.sample_rate, wav)
            print("WAV file saved", 'gaze_utils/audio.wav')


# Recorder('gaze_utils/audio.wav').record()

# import subprocess
# subprocess.run('/home/corallab/anaconda3/envs/rmp_cograsp/bin/python -m whisper audio.wav --model small', shell=True)
