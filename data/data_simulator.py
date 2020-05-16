import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo

np.random.seed(9345)

data_tickets = pd.DataFrame({'size':[0,500,1000,3000],
                             'price':[0,900,1500,3500]})

n_data = 10

data_usage = np.concatenate([np.concatenate([np.linspace(3500,500,15+np.random.randint(low=-5,high=5)),
                                             #np.linspace(1500,500,15+np.random.randint(low=-5,high=5))
                                            ]
                                           ) for _ in range(n_data)]).ravel().tolist()

data_usage = data_usage - abs(np.random.randint(low=0,high=75,size=len(data_usage)))

test_data_usage = np.concatenate([np.concatenate([np.linspace(3500,500,15+np.random.randint(low=-5,high=5)),
                                             #np.linspace(1500,500,15+np.random.randint(low=-5,high=5))
                                            ]
                                           ) for _ in range(n_data)]).ravel().tolist()

test_data_usage = test_data_usage - abs(np.random.randint(low=0,high=75,size=len(test_data_usage)))

fig = go.Figure(data=go.Scatter(x=list(range(len(data_usage))),
                                y=data_usage,
                                hovertemplate='<i>Day</i>: %{x:.d}<br>'+\
                                              '<b>Remaining data</b>: %{y:.1f}',
                                name='Simulated train data'))

fig.add_trace(go.Scatter(x=list(range(len(test_data_usage))),
                         y=test_data_usage,
                         hovertemplate='<i>Day</i>: %{x:.d}<br>'+\
                                              '<b>Remaining data</b>: %{y:.1f}',
                         name='Simulated test data'))

fig.layout.update(title='Simulated sample data',
                  xaxis=dict(title='Days'),
                  yaxis=dict(title='Data usage in MB'),)


data_tickets.to_csv('data_tickets.csv',index=False)
np.savetxt('data_usage.csv', data_usage, delimiter=',')
np.savetxt('test_data_usage.csv', test_data_usage, delimiter=',')

pyo.plot(fig,filename='../images/sample_data.html')