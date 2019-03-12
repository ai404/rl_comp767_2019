import pendulum as pd
import numpy as np
import random
import matplotlib.pyplot as plt 
import functions as fun
import matplotlib.animation
from matplotlib.colors import LinearSegmentedColormap
import pickle


def main():
	visualisePend=False

	t=0.7
	episodes=200
	steps=200
	lambdas=[0,0.3,0.7,0.9,1]
	alphas=[1/4,1/8,1/16]
	runs=np.round(np.random.rand(10)*10000,0)

	state_0=[0,0]
	data=[]

	for lamb in lambdas:
		print('Processing lambda:' + str(lamb))
		data_lambda=[]
		for alp in alphas:
			print('\tProcessing alpha:'+ str(alp))
			data_alpha=[]
			for s in range(len(runs)):
				print('\t\tCompleting run ' +str(s+1) +' of ' + str(len(runs)))
				data_run=np.zeros((episodes,1))
				tiles=fun.Tiles(state_0,lamb,alp,int(runs[s]),[-np.pi,np.pi],[-8,8])
				for e in range(episodes):
					tiles.initialiseET()
					pend=pd.PendulumEnv()
					pend.reset()
					pend.state=state_0
					theta, v = pend.state
					pos=np.zeros((steps+1,2))
					pos[0,:]=[0,-1]
					data_episode=np.zeros((steps+1,1))
					data_episode[0]=tiles.get_value(state_0)
					for i in range(steps):
						torque=fun.determineTorque(v,t)
						obs=pend.step([torque])
						reward=obs[1]
						theta, v = pend.state
						theta=pd.angle_normalize(theta)
						tiles.updateValues(theta,v,reward)
					data_run[e]=tiles.get_value(state_0)
				data_alpha.append(data_run)
			data_lambda.append(data_alpha)
		data.append(data_lambda)
		
	with open('data_2.pkl', 'wb') as handle:
		pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

	## to visualize
	if visualisePend:
		fig, ax = plt.subplots()
		ax.set_xlabel('X Axis', size = 12)
		ax.set_ylabel('Y Axis', size = 12)
		ax.axis([-1,1,-1,1])
		x_vals = []
		y_vals = []
		intensity = []
		iterations = 200
		
		t_vals = np.linspace(0,1*iterations, iterations)

		colors = [[0,0,1,0],[0,0,1,0.5],[0,0.2,0.4,1]]
		cmap = LinearSegmentedColormap.from_list("", colors)
		scatter = ax.scatter(x_vals,y_vals, c=[], cmap=cmap, vmin=0,vmax=1)
		
		count=0
		def get_new_vals():
			global count
			x=pos[count,0]
			y=pos[count,1]
			count+=1
			return [x], [y]
		
		def update(t):
			global x_vals, y_vals, intensity
			# Get intermediate points
			new_xvals, new_yvals = get_new_vals()
			x_vals.extend(new_xvals)
			y_vals.extend(new_yvals)

			# Put new values in your plot
			scatter.set_offsets(np.c_[x_vals,y_vals])
			
			#calculate new color values
			intensity = np.concatenate((np.array(intensity)*0.96, np.ones(len(new_xvals))))
			scatter.set_array(intensity)
			
			# Set title
			ax.set_title('Time: %0.3f' %t)

		ani = matplotlib.animation.FuncAnimation(fig, update, frames=t_vals,interval=50)
		plt.show()

	tiles.showTilings()

if __name__ == "__main__":
	main()

print('done')
#pend.env.close()#