from dependency import *
from GeneticAlgorithm import GeneticAlgorithm
from EvaluateFunctions import EvaluateFunctions

funcs = EvaluateFunctions()

def write(path, info):
    with open(path,"w") as f:
        f.write(info)

def writeMatrix(path, matrix):
    info = ""
    with open(path,"w") as f:
        for line in matrix:
            for num in line:
                info+=str(num) + " "
            info += "\n"
        f.write(info)

# 根据输入的func，测试并绘制进化算法的搜索效果
def drawGeneticTrace(num):
    y = np.empty(0, dtype=float)
    func = funcs.getEvaluateFunc(num)
    algorithm = getAlgorithmOfFunc(num)
    fa = algorithm.initPopulation()
    fa.sort(key=func)
    y = np.append(y, func(fa[0]))
    for i in range(algorithm.GENERATION):
        children = algorithm.getChildPopulation(fa=fa)
        total = fa + children
        total.sort(key=func)
        fa = total[:algorithm.POP_SIZE]
        y = np.append(y, func(fa[0]))
    print(fa[0])
    print(func(fa[0]))

    x = np.arange(algorithm.GENERATION + 1)
    plt.plot(x,y)
    plt.show()

def getAlgorithmOfFunc(funcIndex):
    if funcIndex==1:
        return GeneticAlgorithm(-100,100,30)
    if funcIndex==2:
        return GeneticAlgorithm(-10,10,30)
    if funcIndex==3:
        return GeneticAlgorithm(-100,100,30)
    if funcIndex==4:
        return GeneticAlgorithm(-100,100,30)
    if funcIndex==5:
        return GeneticAlgorithm(-30,30,30)
    if funcIndex==6:
        return GeneticAlgorithm(-100,100,30)
    if funcIndex==7:
        return GeneticAlgorithm(-1.28,1.28,30)
    if funcIndex==8:
        return GeneticAlgorithm(-500,500,30)
    if funcIndex==9:
        return GeneticAlgorithm(-5.12,5.12,30)
    if funcIndex==10:
        return GeneticAlgorithm(-32,32,30)
    if funcIndex==11:
        return GeneticAlgorithm(-600,600,30)
    if funcIndex==12:
        return GeneticAlgorithm(-50,50,30)
    if funcIndex==13:
        return GeneticAlgorithm(-50,50,30)
    if funcIndex==14:
        return GeneticAlgorithm(-65.53,65.53,2)
    if funcIndex==15:
        return GeneticAlgorithm(-5,5,4)
    if funcIndex==16:
        return GeneticAlgorithm(-5,5,2)
    if funcIndex==17:
        return GeneticAlgorithm(-5,15,2)
    if funcIndex==18:
        return GeneticAlgorithm(-2,2,2)
    if funcIndex==19:
        return GeneticAlgorithm(0,1,3)
    if funcIndex==20:
        return GeneticAlgorithm(0,1,6)
    if funcIndex==21:
        return GeneticAlgorithm(0,10,4)
    if funcIndex==22:
        return GeneticAlgorithm(0,10,4)
    if funcIndex==23:
        return GeneticAlgorithm(0,10,4)

# 23个基准测试函数，每个跑times遍，同时存储数据
def multiRun23TestFuncOfGA(times):
    for i in range(1,24):
        print("test function: " + str(i) + " begin")
        algorithm = getAlgorithmOfFunc(i)
        # 保存N次实验的数据，每一行代表第i代的最优解
        x = np.arange(algorithm.GENERATION + 1)
        matrix = []
        for j in range(times):
            random.seed(j)
            print(j)
            y = []
            fa = algorithm.initPopulation()
            func = funcs.getEvaluateFunc(i)
            fa.sort(key= func)
            y.append(func(fa[0]))
            for k in range(algorithm.GENERATION):
                children = algorithm.getChildPopulation(fa)
                total = fa + children
                total.sort(key=func)
                fa = total[:algorithm.POP_SIZE]
                y.append(func(fa[0]))
            matrix.append(y)
        writeMatrix("data\\GA\\func_"+str(i)+"_data.txt",matrix)
        data = np.array(matrix)
        mean = np.average(data, axis=0)
        info = ""
        for num in mean:
            info += str(num) + " "
        write("data\\GA\\func_"+str(i)+"_avg.txt", info)
        plt.title("Test Function " + str(i) + " with GA (avg of " + str(times) +" times)")
        plt.plot(x,mean)
        plt.savefig("data\\GA\\func_"+str(i)+"_avg.png")
        plt.clf()
        # plt.show()

            
            
multiRun23TestFuncOfGA(30)


# drawGeneticTrace(23)

        
        


