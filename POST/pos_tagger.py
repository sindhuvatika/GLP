import copy
import pickle
import collections

class POS_Tagger:

	def __init__(self):
		
		self.count_tags=dict()	#c(t[i])
		self.global_data=[]		#c(w[i],t[i])
		self.bigram_tags=[]		#c(t[i-1],t[i])
		self.vocab_global_data=set()
		self.n=2
		self.tags=[]
		self.lit=[]
		self.tagset=set(self.tags)


	def read_data(self,filename):
		
		with open(filename,'r') as file:
			for i in file.read().split('\n'):
				x=i.split("\t")
				temp=x.pop(0)
				del temp
				self.lit.append(x[0])

	def pos_tagger_train(self,filename):

		self.read_data(filename)
		for i in self.lit:
			i=i.split()
			temp=[]
			for j in i:
				new_temp=tuple(j.split("|"))
				self.tags.append(new_temp[1])
				self.global_data.append(new_temp)

		self.count_tags=dict(collections.Counter(self.tags))
		self.global_data=dict(collections.Counter(self.global_data))
		self.vocab_global_data=set(key for key in self.global_data.keys())
		for i in range(len(self.tags)):
			temp=''
			try:
				for j in range(self.n):
					temp+=" "+self.tags[i+j]
			except IndexError:
				continue
			self.bigram_tags.append(tuple(temp.lstrip().split()))
		self.bigram_tags=dict(collections.Counter(self.bigram_tags))
		#print(vocab_global_data)
		#print('count_tags : ',count_tags.keys(),end='\n')
		#print('global_data : ',global_data,end='\n')
		#print('bigram_tags : ',bigram_tags,end='\n')
		#print("\n\n")

	def Pt_t_1(self,tag,tag_1):
		if (tag_1,tag) not in list(self.bigram_tags.keys()):
			self.bigram_tags[(tag_1,tag)]=0
		temp=(self.bigram_tags[(tag_1,tag)]/self.count_tags[tag_1])
		return temp


	def Pw_t(self,word,tag):
		if (word,tag) not in list(self.global_data.keys()):
			self.global_data[(word,tag)]=0
		temp=(self.global_data[(word,tag)]/self.count_tags[tag])
		return temp


	def Ptag(self,tag):
		temp=(self.count_tags[tag]/len(self.tags))
		return temp


	def keyWithMaxValue(self,p_tags):
		key=list(p_tags.keys())
		value=list(p_tags.values())
		return key[value.index(max(value))]


	def predict(self,input_sentence):
		self.tagset=set(self.tags)
		assigned_tags=[]
		count=0
		for i,word in enumerate(input_sentence):

			P_tags=dict()
			if i==0:
				for tag in self.tagset:
					P_tags[tag]=self.Ptag(tag) * self.Pw_t(word,tag)
			else:
				for tag in self.tagset:
					P_tags[tag]=self.Pt_t_1(tag,assigned_tags[i-1])*self.Pw_t(word,tag)
				
			assigned_tags.append(self.keyWithMaxValue(P_tags))
			if P_tags[self.keyWithMaxValue(P_tags)]==0:
				count+=1
			if count>=1:
				break
			#print(word,P_tags,sep=" : ",end="\n")

		if count>=1:
			print("Sentence is grammatically incorrect!")
			print("OR")
			print("Insufficient training data is provided!")

		else:
			print(input_sentence,end='\n')
			print(assigned_tags,end='\n')
	
	def write_weights(self):
		
		with open('weights/global_data.encode','wb') as file:
			pickle.dump(self.global_data,file)

		with open('weights/bigram_tags.endoce','wb') as file:
			pickle.dump(self.bigram_tags,file)

		with open('weights/count_tags.encode','wb') as file:
			pickle.dump(self.count_tags,file)

		with open('weights/tags.encode','wb') as file:
			pickle.dump(self.tags,file)

	def read_weights(self):
		
		with open('weights/global_data.encode','rb') as file:
			self.global_data=pickle.load(file)
		
		with open('weights/bigram_tags.encode','rb') as file:
			self.bigram_tags=pickle.load(file)
		
		with open('weights/count_tags.encode','rb') as file:
			self.count_tags=pickle.load(file)

		with open('weights/tags.encode','rb') as file:
			self.tags=pickle.load(file)


#filename='datasets/s1'
pos=POS_Tagger()
#pos.pos_tagger_train(filename)
pos.read_weights()
print("\n")
pos.predict('અમદાવાદ ભારતનું ગુજરાતનું સૌથી મોટું શહેર છે'.split())
print("\n")
pos.predict('મોર ભારતનો રાષ્ટ્રીય પક્ષી છે'.split())
print("\n")
