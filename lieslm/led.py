import Jetson.GPIO as GPIO
import warnings
# Avoid warnings from GPIO:
GPIO.setwarnings(False)
warnings.filterwarnings("ignore", message="Could not open /dev/mem")
import time

BLUE = "\033[94m"
RESET = "\033[0m"


def blink_led(duration_seconds, pin=7): #pin 7 is "aud" in Nvidia's world 
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
    start_time = time.time()
    end_time = start_time + duration_seconds

    print(f"{BLUE}[+] Blinking on pin {pin} for {duration_seconds}s...{RESET}")
    while time.time() < end_time:
        remaining_ratio = (end_time - time.time()) / duration_seconds
        
        delay = max(0.05, 0.5 * remaining_ratio)
        
        GPIO.output(pin, GPIO.HIGH)
        time.sleep(delay)
        GPIO.output(pin, GPIO.LOW)
        time.sleep(delay)
        GPIO.output(pin, GPIO.HIGH)

def clean_led(pin=7):
    GPIO.output(pin, GPIO.LOW)
    GPIO.cleanup()

# for testing, because GPIO usage on Jetpack 6.x is awful ! Thanks NVIDIA !
if __name__ == "__main__":
    blink_led(10)
    time.sleep(5)
    clean_led()
