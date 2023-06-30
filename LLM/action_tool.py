from typing import List, Union
import io
import time
from datetime import datetime

import langchain
from langchain.agents import Tool


class ActionTool(Tool):
    def __init__(
        self,
        name:str,
        description:str,
        model_comm_filepath:str
    ):
        Tool.__init__(
            self,
            name=name,
            func=self._func,
            description=description,
        )
        self.model_comm_filepath = model_comm_filepath

    def _func(
        self,
        text:str,
    ):
        """
        :param text: str that contains the input arguments to the action.
            The format may be approximative and left for the model to parse properly.
        """
        now = datetime.now()
        datetime_str = now.strftime("%Y-%m-%d %H:%M:%S")

        str2write = f"{datetime_str} :: {self.name} {text}"
        writing_successful = False
        while not writing_successful:
            try:
                with open(self.model_comm_filepath, 'w') as file:
                    file.write(str2write)
                writing_successful = True
            except Exception as e:
                print(f"WARNING: Exception caught when trying to write:\n '{str2write}'\n ...in file '{self.model_comm_filepath}' :\n {e}")
            time.sleep(0.25)

        observation = None
        while observation is None:
            listening_successful = False
            while not listening_successful:
                try:
                    with open(self.model_comm_filepath, 'r') as file:
                        text = file.read()
                    listening_successful = True
                except Exception as e:
                    print(f"WARNING: Exception caught when trying to read from file '{self.model_comm_filepath}' :\n {e}")
            if text != str2write:
                observation = text
            else:
                time.sleep(0.5)

        return observation


