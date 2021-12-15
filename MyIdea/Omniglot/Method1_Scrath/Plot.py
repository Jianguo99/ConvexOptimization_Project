import pandas as pd
data = pd.read_csv("./Method1_Scrath/Mini.csv")
loss = data.iloc[0,1]
loss = eval(loss)
acc = data.iloc[0,2]
acc = eval(acc)
end_iteration = data.iloc[0,3]
end_iteration =eval(end_iteration)
import matplotlib.pyplot as plt
plt.plot(loss)
plt.show()
plt.plot(acc)
plt.show()
plt.plot(end_iteration)
plt.show()