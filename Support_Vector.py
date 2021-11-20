
import numpy as np
import tensorflow as tf
import seaborn as sns 
import matplotlib.pyplot as plt
np.random.seed(10)
# %matplotlib inline

x1 = np.random.normal(5,1,500)
x2 = np.random.normal(6,1,500)
x3 = np.random.normal(10,1,500)
x4 = np.random.normal(12,1,500)

x = np.stack([x1,x2],axis = 1)
x13 = np.concatenate([x1,x3],axis = 0)
x24 = np.concatenate([x2,x4],axis = 0)

x = np.stack([x13,x24],axis = 1)

y = [int(a[0] >=min(x3) and a[1] >=min(x4)) for a in x]
y = np.array(y)
y[y == 0] = -1

# y = tf.squeeze(tf.nn.sigmoid(y)
pos = x[y==1]
neg = x[y==-1]

fig = plt.figure(figsize = (20,15))
sns.scatterplot(pos[:,0],pos[:,1],pos[:,2],label='pos',color = 'red')
sns.scatterplot(neg[:,0],neg[:,1],label='neg',color = 'blue')
plt.show()

x = tf.cast(x,dtype = tf.float32)
# w = tf.random.normal(shape = (2,1),dtype = tf.float32)
# final = tf.linalg.matmul(x,w)

def model():
  inputs = tf.keras.layers.Input(shape = (2,))
  outputs = tf.keras.layers.Dense(1)(inputs)

  model = tf.keras.Model(inputs = inputs,outputs = outputs)
  model.compile(loss = 'hinge',optimizer = 'adam')
  return model

m = model()
m.fit(x,y,epochs = 1000)

print(m.get_weights())

pred = m.predict(x)
print(pred.shape,x.shape)

sv = []
for ind,elem in enumerate(pred):
  if elem >= -1 and elem <= 1:
    sv.append(x[ind])


sv = np.array(sv)
print(sv)
sv = np.concatenate([sv,np.zeros((1000 - len(sv),2))],axis = 0)
print(sv,sv.shape)

fig = plt.figure(figsize = (20,15))
sns.scatterplot(pos[:,0],pos[:,1],label='pos',color = 'orange')
sns.scatterplot(neg[:,0],neg[:,1],label='neg',color = 'blue')
sns.scatterplot(sv[:,0],sv[:,1],label='sv',color = 'red')
plt.show()

