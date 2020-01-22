import plotly.graph_objects as go
import numpy as np
class figure(object):

    def __init__(self):
        self._data = {}

    def atom(self, coordinate, mass, name='atom'):
        self._data['atom'] = go.Scatter3d(
            x=coordinate[0], 
            y=coordinate[1], 
            z=coordinate[2], 
            mode='markers',
            marker=dict(
                symbol='circle',
                size=mass*0.5,
                color=coordinate,
                colorscale='Viridis',
                line=dict(color='rgb(50,50,50)', width=0.5)
                ),
            name=name, )

    def box(self, box):
        rotate = lambda i: i if i<3 else i-3
        lines = []
        for i in range(3):
            for p in [np.array([0,0,0]), np.array(box[i])]:
                lines.append(p)
                lines.append(lines[-1]+np.array(box[rotate(i+1)]))
                lines.append(lines[-1]+np.array(box[rotate(i+2)]))
                lines.append(lines[-1]-np.array(box[rotate(i+1)]))
                lines.append(lines[-1]-np.array(box[rotate(i+2)]))
        self._data['box'] = go.Scatter3d(
            x=np.array(lines)[:,0],
            y=np.array(lines)[:,1], 
            z=np.array(lines)[:,2],
            mode='lines',
            name='box',
            line=dict(
                color='darkblue',
                width=4
                ))
            
    def velocity(self, coordinate, velocity):
        self._data['velocity'] = go.Cone(
            x=coordinate[0], 
            y=coordinate[1], 
            z=coordinate[2], 
            u=velocity[0],
            v=velocity[1],
            w=velocity[2],
            colorscale='Blues',
            sizemode="absolute",
            sizeref=0.01, )

    def show(self):
        fig = go.Figure(data=list(self._data.values()))
        fig.update_layout(
            #width=800,
            #height=800,
            #autosize=False,
            scene=dict(
                aspectratio = dict(x=1, y=1, z=1),
                aspectmode = 'manual'
            ))

        fig.show()
    


