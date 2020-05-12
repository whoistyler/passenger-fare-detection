import json
import urllib.request as request
import urllib.error
import datetime
import threading

API_URL = "http://192.168.43.219:5000"

def http_req(url, params=None):
    data = json.dumps(params).encode('utf8') if params else None
    req = request.Request(url, data=data,
                          headers={'content-type': 'application/json'})
    try:
        res = request.urlopen(req)
        return res.read().decode('utf8')
    except urllib.error.URLError as err:
        print("[ERROR] http_req error: " + str(err))
        return None


STOP_IDS = [3536, 3684, 3685, 3677]
BUS_ID = "SGDH612"
ROUTE_ID = "31"

class BusState:
    def __init__(self):
        self.stop_num = 0

        # aggregated data
        self.current_riders = 0
        self.evaders_on_board = 0

        # per stop data
        self.entrants = 0
        self.exiters = 0
        self.suss_entrants = 0
        self.suss_exiters = 0
        self.curr_suss_count = 0

    def __str__(self):
        return str(vars(self))

    def enter(self):
        """ Mark a passenger as entering the bus """
        self.entrants += 1

    def enter_suss(self):
        """ Mark a suspected evader as entering the bus """
        self.suss_entrants += 1

    def exit(self):
        """ Mark a passenger as exiting the bus """
        self.exiters += 1

    # This should store the person ID, the stop they entered and the stop that they exited off of for analytics
    def exit_suss(self):
        """ Mark a suspected evader as exiting the bus """
        self.suss_exiters += 1

    def next_stop(self):
        """
        Update bus to next stop. Sends previous stop information to server.
        """
        self.current_riders += self.entrants - self.exiters
        self.evaders_on_board += self.suss_entrants - self.suss_exiters
        now = datetime.datetime.now()
        formatted_now = now.strftime("%a, %d %b %Y %H:%M:%S %Z") ### Might need to use this

        stop_info = {
            "bus_license_num": BUS_ID,
            "route_id": ROUTE_ID,
            "stop_id": STOP_IDS[self.stop_num],
            "suspects_entered": self.suss_entrants,
            "suspects_exited": self.suss_exiters,
            "time_of_entry": formatted_now,
            "total_entered": self.entrants,
            "total_exited": self.exiters,
            "total_passengers": self.current_riders,
            "total_suspects": self.curr_suss_count
        }

        self.stop_num = (self.stop_num + 1) % len(STOP_IDS)
        self.entrants = 0
        self.exiters = 0
        self.suss_entrants = 0
        self.suss_exiters = 0

        print(f"[INFO] Sending data to server:\n{stop_info}")
        call = API_URL + '/backend/stop_info', stop_info
        http_call = threading.Thread(target=http_req, args=call)
        http_call.start()

if __name__ == "__main__":
    b = BusState()

    # Use this to test backend

    # b.enter()
    # b.enter()
    # b.enter()
    # b.enter()
    # b.enter_suss()

    # b.next_stop()

    # b.enter()
    # b.exit()
    # b.enter_suss()

    # b.next_stop()

    # b.enter()
    # b.exit()
    # b.exit_suss()

    # b.next_stop()

    # b.enter()
    # b.exit()

    # b.next_stop()

    ##print(f"bus state: {b}")
    # print("GET /backend/stop_info")
    # call = API_URL + "/backend/stop_info", None
    # http_call = threading.Thread(target=http_req, args=call)
    # http_call.start()
