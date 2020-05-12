import threading
import socket
import sys
# import netifaces as ni // ipv4 address 

class PaymentServerThread(threading.Thread):
    """This class listens to IDs passed from the simulated payment terminals

    Notes:
        Set HOST by running ipconfig and pasting the value for "IPv4 Address:"
        Make sure PORT is the same as the port on the Message_Sender class within the simulated Android payment terminals

    Attributes:
        self.__HOST (str): The local host on which the machine is running on, use ipconfig to configure to your system
        self.__PORT (str): The port which your socket will be initialized on
    """
    # we could use this if wifi0 is the dev name ni.ifaddresses('wifi0')[ni.AF_INET][0]['addr']
    __HOST = '192.168.0.24'     # this is your localhost
    __PORT = 8888            # your desired port which is NOT being used by other processes
    __is_running = False
    payment_id_queue = None

    def __init__(self, payment_id_queue):
        """Starts up the PaymentServer

        Args:
            payment_id_queue: The queue passed from PersonTracker used to
        """
        threading.Thread.__init__(self)
        self.payment_id_queue = payment_id_queue

    def run(self):
        """Creates socket at self.PORT on local machine

        Returns:
            None
        """
        self.__is_running = True
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # socket.socket: must use to create a socket.
        # socket.AF_INET: Address Format, Internet = IP Addresses.
        # socket.SOCK_STREAM: two-way, connection-based byte streams.

        # Bind socket to Host and Port
        try:
            s.bind((self.__HOST, self.__PORT))
        except socket.error as err:
            print('Bind Failed, Error Code: ' + str(err[0]) + ', Message: ' + err[1])
            sys.exit()

        s.listen(10)  # set up and start TCP listener.
        print('[INFO] Socket Initialized and Server Listening')

        while self.__is_running:
            # sleep until a new msg is read on the socket
            print('listening...')
            conn, addr = s.accept()

            # For debugging signals
            # print('Connect with ' + addr[0] + ':' + str(addr[1]))

            buf = conn.recv(64)  # In bytes
            payment_id = buf.decode('utf-8')  # Convert bytes to a Unicode Text string
            print(payment_id)

            self.payment_id_queue.put(payment_id)

        s.close()  # close the socket

    def stop(self):
        """ Stop the thread

        We are going to connect to the socket to cause the while loop to continue running and check the condition
        __is_running which will be false

        Returns:
            None
        """
        self.__is_running = False
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((self.__HOST, self.__PORT))
