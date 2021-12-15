from ipykernel.comm import Comm

class SmartEDA:

    def __init__(self):
        self.my_comm = Comm(target_name='my_comm_target', data={'foo': 1})

        @self.my_comm.on_msg
        def _recv(msg):
            # Use msg['content']['data'] for the data in the message
            print(msg)

    # Comm class sends only serializable data, e.g. arrays, numbers, strings et al.
    #     For dataframe, we can do a trick like df.to_dict(orient='records'), and
    #     re-assemble it at front-end
    def init_dataframe(self, df):
        try:
            self.my_comm.send({'dataframe': df.to_dict(orient='records')})
            self.df = df
        except:
            print('Error: failed during dataframe initialization, try again')
