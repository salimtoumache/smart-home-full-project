import os
import glob
import serial
import time
ser = serial.Serial("/dev/ttyUSB0", 9600)
while True:
    datasend=input('entry: ')
    datasend=datasend+"\r"
    ser.write(datasend.encode())
