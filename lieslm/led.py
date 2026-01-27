import Jetson.GPIO as GPIO
import time

def blink_led(duration_seconds, pin=7):
    # Pin Setup
    GPIO.setmode(GPIO.BOARD) # Use physical pin numbering
    GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
    
    start_time = time.time()
    end_time = start_time + duration_seconds
    

    print(f"Blinking on pin {pin} for {duration_seconds}s...")
    while time.time() < end_time:
        # Calculate how much time is left (from 1.0 down to 0.0)
        remaining_ratio = (end_time - time.time()) / duration_seconds
        
        # Adjust the delay based on remaining time
        # Max delay 0.5s, min delay 0.05s
        delay = max(0.05, 0.5 * remaining_ratio)
        
        GPIO.output(pin, GPIO.HIGH)
        time.sleep(delay)
        GPIO.output(pin, GPIO.LOW)
        time.sleep(delay)
        GPIO.output(pin, GPIO.HIGH)

def clean_led(pin=7):
    GPIO.output(pin, GPIO.LOW)
    GPIO.cleanup() # Always clean up to reset pins
    print("Done and cleaned up.")

# Usage: Blink for 10 seconds
if __name__ == "__main__":
    blink_led(10)
    time.sleep(5)
    clean_led()
