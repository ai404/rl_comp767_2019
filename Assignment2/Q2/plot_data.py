import pickle
import numpy as np
import matplotlib.pyplot as plt

def main():
	episodes=200
	steps=200
	lambdas=[0,0.3,0.7,0.9,1]
	alphas=[1/4,1/8,1/16]
	colors=['r','b','g']
	with open('data.pkl', 'rb') as handle:
		data = pickle.load(handle)
		
	x=np.linspace(1,episodes,episodes)
	for i in range(len(data)): ## lambdas
		plt.figure()
		for j in range(len(data[i])): ## alphas
			data_plot=np.zeros((data[0][0][0].shape))
			count=0
			for k in range(len(data[i][j])): ## runs
				data_plot+=data[i][j][k]
				count+=1
			data_plot/=count
			data_inf= 'alpha = '+str(alphas[j])
			plt.plot(x,data_plot,colors[j], label=data_inf)
			
		plt.ylabel('Value of theta = 0, velocity=0')
		plt.xlabel('Episodes')
		plt.legend()
		plt.title('lambda = '+str(lambdas[i])+', 10 runs with 200 episodes')
		plt.savefig('lambda_'+str(lambdas[i])+'.png')
				
					


if __name__ == "__main__":
	main()
