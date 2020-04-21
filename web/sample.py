import subprocess
import signal

def get():
    driver = subprocess.Popen(['./../build/driver'], shell = False)

    print("Afterwards...")
    while (True):
        enter = input("Enter to signal: ")
        driver.send_signal(signal.SIGHUP)
