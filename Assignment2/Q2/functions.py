import numpy as np
import random
import matplotlib.pyplot as plt

def determineTorque(v,s):
	v=np.sign(v)
	decision=random.random()
	if not v==0:
		if decision<=0.9:
			return v*s
		else:
			return v*-1*s
	else:
		return (-1)**random.randrange(2)*s
		

class Tiles:
	def __init__(self,state,lamb,alpha,seed_,x_lim,y_lim,gamma=0.9,weights=[-0.001,0.001], bins=[10,10],tilings=5):
		self.lamb=lamb
		self.alpha=alpha
		np.random.seed(seed_)
		self.x_lim=x_lim
		self.y_lim=y_lim
		self.weights=weights
		self.bins=bins
		self.tilings=tilings
		self.gamma=gamma
		self.values=self.initialiseTiles()
		self.tilesLim=self.initialiseLimits()
		self.ET=self.initialiseET()
		self.prv_state=self.findTiles(state[0],state[1])
		
	def initialiseET(self):
		return np.zeros((self.bins[0],self.bins[1],self.tilings))
	
	def initialiseTiles(self):
		return np.random.rand(self.bins[0],self.bins[1],self.tilings)*np.sum(np.abs(self.weights))+self.weights[0]
		
	def initialiseLimits(self):
		tileLimX=np.zeros((self.tilings,self.bins[0]+1))
		tileLimX[:,0]=self.x_lim[0]
		tileLimX[:,-1]=self.x_lim[1]
		tileLimX[0,1:self.bins[0]]=np.linspace(self.x_lim[0],self.x_lim[1],self.bins[0]+1)[1:self.bins[0]]
		offset=(tileLimX[0,1]-tileLimX[0,0])/5
		for i in range(1,self.tilings):
			tileLimX[i,1:self.bins[0]]=tileLimX[i-1,1:self.bins[0]]+i*offset*(-1)**i
		
		tileLimY=np.zeros((self.tilings,self.bins[1]+1))
		tileLimY[:,0]=self.y_lim[0]
		tileLimY[:,-1]=self.y_lim[1]
		tileLimY[0,1:self.bins[1]]=np.linspace(self.y_lim[0],self.y_lim[1],self.bins[0]+1)[1:self.bins[0]]
		offset=(tileLimY[0,1]-tileLimY[0,0])/5
		for i in range(1,self.tilings):
			tileLimY[i,1:self.bins[0]]=tileLimY[i-1,1:self.bins[0]]+i*offset*(-1)**i
			
		return [tileLimX,tileLimY]
		
	def showTilings(self):
		colors=['r','b','g','k','m']
		for axis in range(len(self.tilesLim)):
			for tiling in range(self.tilings):
				for tile in range(self.tilesLim[axis].shape[1]):
					if axis==0:
						plt.plot(np.ones(2)*self.tilesLim[axis][tiling,tile],self.y_lim,colors[tiling])
					else:
						plt.plot(self.x_lim,np.ones(2)*self.tilesLim[axis][tiling,tile],colors[tiling])
		plt.ylabel('Angular velocity (rad/s)')
		plt.xlabel('Angular position (rad)')
		plt.title('Tilings of the state space')
		# plt.show()
		plt.savefig('tilings.png')
	def findTiles(self,x,y):
		tiles=np.zeros((self.tilings,2))
		for i in range(self.tilings):
			tiles[i,:]=[np.searchsorted(self.tilesLim[0][i,:],x),np.searchsorted(self.tilesLim[1][i,:],y)]
		tiles[tiles[:,0]==self.bins[0],0]=self.bins[0]-1
		tiles[tiles[:,1]==self.bins[1],1]=self.bins[1]-1
		return tiles.astype(int)
		
	def get_value(self,state):
		state=self.findTiles(state[0],state[1])
		value=0
		for i in range(self.tilings):
			value+=self.values[state[i,0],state[i,1],i]
		return value
	
	def updateValues(self,x,y,reward):
		state=self.findTiles(x,y)
		self.ET*=self.lamb*self.gamma
		for i in range(self.tilings):
			self.ET[self.prv_state[i,0],self.prv_state[i,1],i]+=1
			TD_error=reward+self.gamma*self.values[state[i,0],state[i,1],i]-self.values[self.prv_state[i,0],self.prv_state[i,1],i]
			self.values[:,:,i]+=self.alpha/self.tilings*TD_error*self.ET[:,:,i]
		self.prv_state=state