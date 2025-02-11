class StockPrediction:
    def __init__(self, ticker, start_date, validation_date, project_folder, epochs, time_steps, token, batch_size):
        self._ticker = ticker
        self._start_date = start_date
        self._validation_date = validation_date
        self._project_folder = project_folder
        self._epochs = epochs
        self._time_steps = time_steps
        self._token = token
        self._batch_size = batch_size

    def get_ticker(self):
        return self._ticker

    def set_ticker(self, value):
        self._ticker = value

    def get_start_date(self):
        return self._start_date

    def set_start_date(self, value):
        self._start_date = value

    def get_validation_date(self):
        return self._validation_date

    def set_validation_date(self, value):
        self._validation_date = value

    def get_project_folder(self):
        return self._project_folder

    def set_project_folder(self, value):
        self._project_folder = value
    
    def get_epochs(self):
        return self._epochs

    def get_time_steps(self):
        return self._time_steps
        
    def get_token(self):
        return self._token     
    
    def get_batch_size(self):
        return self._batch_size     
