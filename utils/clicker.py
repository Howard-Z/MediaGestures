import pyautogui
from time import sleep
import argparse


def click_n_times(n: int):
    print("Initializing...")
    for i in range(1, n + 1):
        if i < n + 1:
            sleep(2)
            print(f"Beginning next recording in 3 for video {i}...", end='\r')
            sleep(1)
            print(f"Beginning next recording in 2 for video {i}...", end='\r')
            sleep(1)
            print(f"Beginning next recording in 1 for video {i}...", end='\r')
            sleep(1)
            print("                                               ", end='\r')

        print(f"Recording video {i}")
        pyautogui.click()
        print(f"3 seconds left...", end='\r')
        sleep(1)
        print(f"2 seconds left...", end='\r')
        sleep(1)
        print(f"1 second left... ", end='\r')
        sleep(2)
        print("                  ", end='\r')
        pyautogui.click()
        print(f"Video {i} completed.")
    print("Done recording.")
    
    
def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument('num_clicks', type=int, help="number of times to click")
    
    args = parser.parse_args()

    num_clicks = args.num_clicks
    click_n_times(num_clicks)


if __name__ == '__main__':
    main()