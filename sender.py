import socket
import struct
import time
import numpy as np


def send_udp_data(ip, port, data_array, frequency_hz):
    """Sends a NumPy array of doubles as little-endian UDP data at a specified frequency."""

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        print(f"Sending UDP data to {ip}:{port} at {frequency_hz} Hz")

        while True:
            start_time = time.time()

            try:
                packed_data = struct.pack("<6d", *data_array)  # Pack as little-endian doubles
                sock.sendto(packed_data, (ip, port))
                #print("Data sent.") # Optional: print each time data is sent

            except Exception as e:
                print(f"Error sending data: {e}")

            end_time = time.time()
            elapsed_time = end_time - start_time

            target_delay = 1.0 / frequency_hz
            actual_delay = target_delay - elapsed_time

            if actual_delay > 0:
                time.sleep(actual_delay)
            #else:
            #    print("Warning: Could not maintain desired frequency.")

    except socket.error as e:
        print(f"Socket error: {e}")
    except KeyboardInterrupt:
        print("Exiting...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'sock' in locals() and sock:
            sock.close()
            print("Socket closed.")


if __name__ == "__main__":
    target_ip = "192.168.1.100"
    target_port = 3827
    frequency = 50  # Hz

    # Example data (replace with your actual data)
    my_data = [0, 0, 0, 0, -90, 0]

    send_udp_data(target_ip, target_port, my_data, frequency)
