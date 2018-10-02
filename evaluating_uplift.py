from numpy import floor, inf, mean, minimum, sign, newaxis, exp
from pandas import DataFrame, merge
from sklearn.linear_model import LinearRegression

class Uplift():
	"""Initializing of data, treatment column name, conversion column name, and uplift column name respectively."""
	
	"""Bibliography:
	Radcliffe NJ. 2012. Moment of Uplift. Technical Note. August 7 2012 Version 2.
	Radcliffe NJ. Using Control Groups to Target on Predicted Lift: Building and Assessing Uplift Models.
	Naranjo OM. 2012. Testing a New Metric for Uplift Models. The University of Edinburgh School of Mathematics."""
	
	
	def __init__(self, mydf, targetcol, outcomecol, upliftcol):
		self.mydf = mydf
		self.targetcol = targetcol
		self.outcomecol = outcomecol
		self.upliftcol = upliftcol
		
	
	def act_pred(self, p = 10):
		"""Calculating The Table of Predicting Uplift and Actual Uplift of every percentile
		and Epsilon respectively. p is the number of percentile that you want to use."""
		tmp=self.mydf.sort_values(self.upliftcol,ascending=False)
		tmp["percentile"] =  floor(tmp[self.upliftcol].rank(method='first',ascending = False) / (tmp.shape[0]+1) * p)
		tmp["targetoutcome"] = tmp[self.targetcol]*tmp[self.outcomecol]
		temp = DataFrame(list(range(p))+list(range(p)),columns=['percentile'])
		temp['uplift'] = inf
		temp['uplift_kind'] = 'kind'
		temp['uplift_kind'][:p]='prediction'
		temp['uplift_kind'][p:]='actual'
		epsilon = 0
		for i in range(p):
			ntk = sum(tmp[tmp["percentile"]==i][self.targetcol])
			nck = sum(1-tmp[tmp["percentile"]==i][self.targetcol])
			temp['uplift'][i] = sum(tmp[tmp["percentile"]==i][self.upliftcol])/(sum(tmp[tmp["percentile"]==i][self.targetcol])+sum(1-tmp[tmp["percentile"]==i][self.targetcol]))
			temp['uplift'][p+i] = sum(tmp[tmp["percentile"]==i]['targetoutcome'])/ntk-sum(tmp[(tmp[self.targetcol]==0)&(tmp["percentile"]==i)][self.outcomecol])/nck
			epsilon += abs(temp['uplift'][i]-temp['uplift'][p+i])*(ntk+nck)
		epsilon /= (sum(tmp[self.targetcol])+sum(1-tmp[self.targetcol]))
		return temp, epsilon
	
	
	def qini_coef(self, negatif_effect = True):
		"""Calculating qini coefficient.
		That is the ratio of model curve area to optimal curve area."""
		temp = self.mydf.sort_values(self.upliftcol,ascending = False)
		def area(tmp):
			tmp['RT'] = (tmp[self.targetcol]*tmp[self.outcomecol]).cumsum()
			tmp['RC'] = ((1-tmp[self.targetcol])*tmp[self.outcomecol]).cumsum()
			tmp['NT'] = tmp[self.targetcol].cumsum()
			tmp['NC'] = (1-tmp[self.targetcol]).cumsum()
			tmp['second_part'] = (tmp['RC'] * tmp['NT'] / tmp['NC']).fillna(0)
			tmp['qini'] = tmp['RT'] - tmp['second_part']
			if negatif_effect:
				result = sum(tmp['qini'])
			else:
				result = sum(tmp[tmp['qini']>0]['qini'])
			return result
		secpart = self.mydf[self.mydf[self.targetcol]==0][self.outcomecol].sum() \
			* float(self.mydf[self.mydf[self.targetcol]==1].shape[0])/self.mydf[self.mydf[self.targetcol]==0].shape[0]
		rqini = self.mydf[self.mydf[self.targetcol]==1][self.outcomecol].sum() - secpart
		rd = range(len(self.mydf)) * rqini / len(self.mydf)
		model_area = area(temp)
		random_area = sum(rd)
		temp['perfect'] = temp[self.outcomecol] * temp[self.targetcol] -  temp[self.outcomecol] * (1-temp[self.targetcol])
		temp=temp.sort_values('perfect',ascending = False)
		optimal_area = area(temp)
		return (model_area-random_area)/(optimal_area-random_area)
		
		
	
	def metric(self, p = 10):
		"""Calculating Spearman correlation, spread, negatif_effect, abs(1-slope), Umax,
		Uplift prediction of first percentile, and Uplift actual of first percentile
		respectively. p is the number of percentile that you want to use."""
		from scipy.stats import spearmanr
		temp, epsilon = self.act_pred(p)
		corr, pvalue = spearmanr(temp['uplift'][:p],temp['uplift'][p:])
		spread = max(temp['uplift'][:p])-min(temp['uplift'][p:])
		#negatif_effect = 0
		#for i in range(0,p,2):
		#	negatif_effect += sign(temp['uplift_pred'][i])
		negatif_effect = abs(sum(sign(temp['uplift'][:p])))
		regr = LinearRegression()
		regr.fit(temp['uplift'][:p].values[:,newaxis],temp['uplift'][p:].values)
		slope = abs(1-regr.coef_[0])
		#Qini = sum(temp['uplift'][p:])
		Umax = max(temp['uplift'][p:])
		uplift1p = temp['uplift'][0]
		uplift1a = temp['uplift'][p]
		return corr, spread, negatif_effect, slope, Umax, uplift1p, uplift1a, epsilon#, Qini
		

	def get_hist(self,p = 10):
		""" calucate the uplift for each decile given an uplift prediction column
		by sorting separately between control and target.
		p is the number of percentile that you want to use."""
		control = self.mydf[self.mydf[self.targetcol]==0]
		target = self.mydf[self.mydf[self.targetcol]==1]
		control=control.sort_values(self.upliftcol,ascending=False)
		target=target.sort_values(self.upliftcol,ascending=False)
		control["percentile"] =  floor(control[self.upliftcol].rank(method='first',ascending = False) / (control.shape[0]+1) * p)
		target["percentile"] =  floor(target[self.upliftcol].rank(method='first',ascending = False) / (target.shape[0]+1) * p)
		control = (control.groupby(["percentile"])[self.outcomecol].mean()).reset_index()
		control.columns = ["percentile", 'prob_control']
		target = (target.groupby(["percentile"])[self.outcomecol].mean()).reset_index()
		target.columns = ["percentile", 'prob_target']
		final = merge(control,target, on = "percentile")
		final["uplift"] = final["prob_target"] - final["prob_control"]
		return final
		
		
	def get_hist_cum(self, p = 10, keys = False) :
		""" calucate the cummulative uplift for each decile given an uplift prediction column.
		p is the number of percentile that you want to use while keys is kind of uplift (True or False)"""
		tmp = self.mydf.sort_values(self.upliftcol,ascending=False)
		tmp["percentile"] =  floor(tmp[self.upliftcol].rank(method='first',ascending = False) / (tmp.shape[0]+1) * p)
		final = []
		for val in range(p) :
			t = tmp[tmp['percentile']<=val]
			target = float(t[t[self.targetcol]==1][self.outcomecol].mean())
			control = float(t[t[self.targetcol]==0][self.outcomecol].mean())        
			if keys:
				uplift = (target - control)* t.shape[0]
			else:
				uplift = (target - control)
			final.append({'percentile':val,"uplift":uplift})
		return DataFrame(final)
		
		
	def gains(self):
		"""Calculating Table of NT (Number of target) versus RT (number of conversion)."""
		tmp = self.mydf.sort_values(self.upliftcol,ascending=False)
		tmp['RT'] = (tmp[self.targetcol]*tmp[self.outcomecol]).cumsum()
		tmp['RC'] = ((1-tmp[self.targetcol])*tmp[self.outcomecol]).cumsum()
		tmp['NT'] = tmp[self.targetcol].cumsum()
		tmp['NC'] = (1-tmp[self.targetcol]).cumsum()
		tmp["upc_1"] =    ((tmp['RT']/tmp['NT']).fillna(0) - (tmp['RC']/tmp['NC']).fillna(0)) *(tmp['NC']+tmp['NT'])
		rd_uc = (self.mydf[self.mydf[self.targetcol]==1][self.outcomecol].sum() / float(self.mydf[self.mydf[self.targetcol]==1].shape[0])\
				- self.mydf[self.mydf[self.targetcol]==0][self.outcomecol].sum() / float(self.mydf[self.mydf[self.targetcol]==0].shape[0]))
		rd = range(self.mydf.shape[0])*rd_uc
		tmp = DataFrame({'NT': range(self.mydf.shape[0]), 'Random':rd, 'RT': tmp["upc_1"]})
		return tmp.set_index('NT')
	
	
	def qini_curve(self) :
		"""Calculating table for plotting qini curve"""
		tmp = self.mydf.sort_values(self.upliftcol,ascending = False)
		tmp['RT'] = (tmp[self.targetcol]*tmp[self.outcomecol]).cumsum()
		tmp['RC'] = ((1-tmp[self.targetcol])*tmp[self.outcomecol]).cumsum()
		tmp['NT'] = tmp[self.targetcol].cumsum()
		tmp['NC'] = (1-tmp[self.targetcol]).cumsum()
		tmp['second_part'] = (tmp['RC'] * tmp['NT'] / tmp['NC']).fillna(0)
		tmp['qini'] = tmp['RT'] - tmp['second_part']
		secpart = self.mydf[self.mydf[self.targetcol]==0][self.outcomecol].sum() \
			* float(self.mydf[self.mydf[self.targetcol]==1].shape[0])/self.mydf[self.mydf[self.targetcol]==0].shape[0]
		rqini = self.mydf[self.mydf[self.targetcol]==1][self.outcomecol].sum() - secpart
		rd = range(len(self.mydf)) * rqini / len(self.mydf)
		tmp = DataFrame({'NT': range(self.mydf.shape[0]), 'Random':rd, 'RT': tmp["qini"]})
		return tmp.set_index('NT')
	
	
	def optimal_curve(self) :
		"""Calculating table for plotting the optimal of qini curve"""
		tmp=self.mydf.copy()
		tmp['perfect'] = tmp[self.outcomecol] * tmp[self.targetcol] -  tmp[self.outcomecol] * (1-tmp[self.targetcol])
		tmp=tmp.sort_values('perfect',ascending = False)
		tmp['RT'] = (tmp[self.targetcol]*tmp[self.outcomecol]).cumsum()
		tmp['RC'] = ((1-tmp[self.targetcol])*tmp[self.outcomecol]).cumsum()
		tmp['NT'] = tmp[self.targetcol].cumsum()
		tmp['NC'] = (1-tmp[self.targetcol]).cumsum()
		tmp['second_part'] = (tmp['RC'] * tmp['NT'] / tmp['NC']).fillna(0)
		tmp['qini'] = tmp['RT'] - tmp['second_part']
		secpart = self.mydf[self.mydf[self.targetcol]==0][self.outcomecol].sum() \
			* float(self.mydf[self.mydf[self.targetcol]==1].shape[0])/self.mydf[self.mydf[self.targetcol]==0].shape[0]
		rqini = self.mydf[self.mydf[self.targetcol]==1][self.outcomecol].sum() - secpart
		rd = range(len(self.mydf)) * rqini / len(self.mydf)
		tmp = DataFrame({'NT': range(self.mydf.shape[0]), 'Random':rd, 'RT': tmp["qini"]})
		return tmp.set_index('NT')
		
	
	def Q_Athey(self, alpha = 0.5):
		"""Calculating expectation of Q according to Athey.
		Alpha is the expected number of Q."""
		tmp = self.mydf.copy()
		tmp['Yi_star'] = tmp[self.outcomecol] * (tmp[self.targetcol]-alpha)/(alpha*(1-alpha)) 
		tmp["delta"] = tmp['Yi_star'] - tmp[self.upliftcol]
		return mean(tmp['delta']*tmp['delta'])
	
	
	def uplift_accent(self, p = 10, miu = None):
		"""Calculating table uplift of each percentile.
		p is the number of percentile that you want to use
		and miu is the overall uplift of sample."""
		tmp = self.mydf.sort_values(self.upliftcol,ascending=False)
		tmp["percentile"] =  floor(tmp[self.upliftcol].rank(method='first',ascending = False) / (tmp.shape[0]+1) * p)
		tmp["targetoutcome"] = tmp[self.targetcol]*tmp[self.outcomecol]
		temp = DataFrame(list(range(p)),columns=['percentile'])
		temp['uplift_pred'] = inf
		temp['uplift_act'] = inf
		temp['uplift_pred_a'] = inf
		temp['uplift_act_a'] = inf
		temp['distance'] = inf
		miu1 = 0
		for i in range(p):
			ukp = sum(tmp[tmp["percentile"]==i][self.upliftcol])/(sum(tmp[tmp["percentile"]==i][self.targetcol])+sum(1-tmp[tmp["percentile"]==i][self.targetcol]))
			miu1 += sum(tmp[tmp['percentile']==i][self.targetcol])*ukp
			ntk = sum(tmp[tmp["percentile"]==i][self.targetcol])
			nck = sum(1-tmp[tmp["percentile"]==i][self.targetcol])
			temp['uplift_pred'][i] = sum(tmp[tmp["percentile"]==i][self.upliftcol])/(sum(tmp[tmp["percentile"]==i][self.targetcol])+sum(1-tmp[tmp["percentile"]==i][self.targetcol]))
			temp['uplift_act'][i] = sum(tmp[tmp["percentile"]==i]['targetoutcome'])/ntk-sum(tmp[(tmp[self.targetcol]==0)&(tmp["percentile"]==i)][self.outcomecol])/nck
		miu1 /= sum(tmp[self.targetcol])
		if miu == None:
			miu = miu1
		temp['uplift_pred_a'] = temp['uplift_pred']-miu
		temp['uplift_act_a'] = temp['uplift_act']-miu
		temp['distance'] = abs(temp['uplift_pred']-temp['uplift_act'])
		return temp
		
	
	def uplift_moment(self, p = 10, orde = None, kind = None, alpha = 0.5, miu = None):
		"""Calculating moment of uplift. Orde can be chosen among "None", "quadratic",
		and "linear". Meanwhile kind can be chosen among "None", "mean", and "minimum"
		if orde is not None. p is the number of percentile that you want to use, 
		while alpha (< 1) is a weigthing penalty term for actual uplift if kind = 'mean',
		and	miu is the overall uplift of sample if orde = None."""
		if orde == None:
			temp = self.uplift_accent(p= p, miu = miu)
			tmp = self.mydf.sort_values(self.upliftcol,ascending=False)
			tmp["percentile"] =  floor(tmp[self.upliftcol].rank(method='first',ascending = False) / (tmp.shape[0]+1) * p)
			Nt = sum(tmp[self.targetcol])
			result = 0
			for i in range(p):
				result += sum(tmp[tmp["percentile"]==i][self.targetcol])*temp['uplift_pred_a'][i]*temp['uplift_act_a'][i]
			result /= Nt	
		elif (orde == 'quadratic' )|(orde == 'linear'):	
			temp = self.uplift_accent(p)
			Nt = sum(self.mydf[self.targetcol])
			tmp = self.mydf.sort_values(self.upliftcol,ascending=False)
			tmp["percentile"] =  floor(tmp[self.upliftcol].rank(method='first',ascending = False) / (tmp.shape[0]+1) * p)
			moment = DataFrame()
			if orde == 'quadratic':
				if kind == None:
					moment['sigma1'] = (minimum(temp['uplift_pred_a'],temp['uplift_act_a'])-abs(temp['uplift_pred_a']-temp['uplift_act_a']))
					moment['sigma1'] = (moment['sigma1'])**2*sign(moment['sigma1'])
					moment['sigma2'] = -(temp['uplift_pred_a']-temp['uplift_act_a'])**2
				elif kind == 'mean':
					moment['sigma2'] = -abs(temp['uplift_pred_a']-temp['uplift_act_a'])**2
					moment['sigma1'] = abs(temp['uplift_pred_a']*(1-alpha)+temp['uplift_act_a']*alpha) - abs(temp['uplift_pred_a']-temp['uplift_act_a'])
					moment['sigma1'] = sign(moment['sigma1'])*(moment['sigma1'])**2
				elif kind == 'minimum':
					moment['sigma1'] = minimum(temp['uplift_pred_a'],temp['uplift_act_a'])-abs(temp['uplift_pred_a']-temp['uplift_act_a'])
					moment['sigma1'] = sign(moment['sigma1'])*(moment['sigma1'])**2
					moment['sigma2'] = -(temp['uplift_pred_a']-temp['uplift_act_a'])
					moment['sigma2'] = sign(moment['sigma2'])*(moment['sigma2'])**2
			elif orde == 'linear':
				if kind == None:
					moment['sigma1'] = abs(minimum(temp['uplift_pred_a'],temp['uplift_act_a'])-abs(temp['uplift_pred_a']-temp['uplift_act_a']))
					moment['sigma2'] = -abs(temp['uplift_pred_a']-temp['uplift_act_a'])
				elif kind == 'mean':
					moment['sigma2'] = -abs(temp['uplift_pred_a']-temp['uplift_act_a'])
					moment['sigma1'] = abs(temp['uplift_pred_a']*(1-alpha)+temp['uplift_act_a']*alpha) - abs(temp['uplift_pred_a']-temp['uplift_act_a'])
				elif kind == 'minimum':
					moment['sigma1'] = minimum(temp['uplift_pred_a'],temp['uplift_act_a'])-abs(temp['uplift_pred_a']-temp['uplift_act_a'])
					moment['sigma2'] = -(temp['uplift_pred_a']-temp['uplift_act_a'])
			result=0
			for i in range(p):
				nt = sum(tmp[tmp['percentile']==i][self.targetcol])
				if temp['uplift_pred_a'][i]*temp['uplift_act_a'][i]>0:
					result += nt*moment['sigma1'][i]
				else:
					result += nt*moment['sigma2'][i]			
			result /= Nt
		return result
	
	
	
	def composite_measures(self, p = 10, tau = 5, negatif_effect = True):
		"""Calculating Family of M. The result contains M1, M2, M3, M4, and M5.
		p is the number of percentile that you want to use, while tau is a parameter for M5."""
		temp = self.act_pred(p)[0]
		corr, spread, negatif_effect, slope, Umax, uplift1p, uplift1a, epsilon = self.metric(p)
		M1 = max(temp['uplift'][p:])/(1+epsilon)
		M2 = corr*M1
		M3 = spread/temp['uplift'][0]*M2
		Qini= self.qini_coef(negatif_effect)
		M4 = Qini/(epsilon+1)
		M5 = Qini*exp(-tau*(slope)**2)
		return M1,M2,M3,M4,M5
	