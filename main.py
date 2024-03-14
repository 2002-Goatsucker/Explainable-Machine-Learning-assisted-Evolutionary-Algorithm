from dependency import *
from ExperimentalMethods import *

# multiRun23TestFuncOfGA(30)
mean = run30TimesOnAllBbobBenchmark(23, bbobtorch.create_f13(dim=30), -100, 100, 30)
min_value = mean[np.size(mean)-1]
print(min_value)
# plt.ylim(min_value,min_value+1000)
plt.plot(np.arange(401), mean)
plt.show()
# drawGeneticTrace(23)

        
        


