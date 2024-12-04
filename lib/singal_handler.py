# In a new file util/signal_handling.py
import threading

class SignalHandler:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.terminate_flag = False
            return cls._instance
    
    def set_terminate(self):
        self.terminate_flag = True
    
    def should_terminate(self):
        return self.terminate_flag

signal_handler = SignalHandler()