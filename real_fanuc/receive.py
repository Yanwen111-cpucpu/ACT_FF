import socket
import time
import struct
import numpy as np  # For easier array handling (install with: pip install numpy)

def receive_udp_message(local_ip, port, frequency_hz):
    """Receives UDP messages containing 36 doubles (little-endian)."""

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", port))
        sock.settimeout(0.01) # Set a timeout to prevent blocking

        print(f"Listening for UDP messages on {local_ip}:{port} at {frequency_hz} Hz")

        while True:
            start_time = time.time()

            try:
                data, addr = sock.recvfrom(36 * 8)  # 36 doubles * 8 bytes each

                if len(data) != 36 * 8:
                    print(f"Warning: Received incomplete data. Expected {36*8} bytes, got {len(data)} bytes.")
                    continue #Skip processing and wait for next message

                # Unpack the data (36 doubles, little-endian)
                unpacked_data = struct.unpack("<36d", data)  # < for little-endian, 36d for 36 doubles

                # Convert to NumPy array for easier use
                data_array = np.array(unpacked_data[18:24])
                # data_array = data_array * 180 / np.pi

                print(f"Received data from {addr}: {data_array}")

            except socket.timeout:
                print("Timeout")
                pass # No message, continue

            except struct.error as e:
                print(f"Error unpacking data: {e}. Data might be corrupted.")
                continue #Skip processing and wait for next message

            except Exception as e:
                print(f"Error receiving or processing data: {e}")
                continue #Skip processing and wait for next message


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
    local_ip = "192.168.1.177"
    port = 9600
    frequency = 50  # Hz
    receive_udp_message(local_ip, port, frequency)