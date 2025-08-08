import sys
import traceback

def error_message_detail(error_message, error_detail: sys):
    exc_type, exc_obj, exc_tb = error_detail.exc_info()
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        return f"error occured in python script name [{file_name}] line number [{line_number}] error message[{error_message}]"
    else:
        return f"error message[{error_message}]"

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
