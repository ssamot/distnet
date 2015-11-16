import numpy as np

from config import experiment_scores_path

task_names_readable  = [
            "0_tasknames",
            "No Metadata",
            "Task Number/Name",
            "QA1 - Single Supporting Fact",
            "QA2 - Two Supporting Facts",
            "QA3 - Three Supporting Facts",
            "QA4 - Two Arg. Relations",
            "QA5 - Three Arg. Relations",
            "QA6 - Yes/No Questions",
            "QA7 - Counting",
            "QA8 - Lists/Sets",
            "QA9 - Simple Negation",
            "QA10 - Indefinite Knowledge",
            "QA11 - Basic Coreference",
            "QA12 - Conjunction",
            "QA13 - Compound Coreference",
            "QA14 - Time Reasoning",
            "QA15 - Basic Deduction",
            "QA16 - Basic Induction",
            "QA17 - Positional Reasoning",
            "QA18 - Size Reasoning",
            "QA19 - Path Finding",
            "QA20 - Agent's Motivations"]


FB_LSTM_Baseline = ["10_baseline", "No Metadata", "LSTM Baseline",0.5 ,  0.2 ,  0.2 ,  0.61,  0.7 ,  0.48,  0.49,  0.45,  0.64,
        0.44,  0.62,  0.74,  0.94,  0.27,  0.21,  0.23,  0.51,  0.52,
        0.08,  0.91]
MEMNET_LSTM_Baseline = ["11_baseline", "No Metadata", "SoS Memnet",1.  ,  1.  ,  1.  ,  1.  ,  0.98,  1.  ,  0.85,  0.91,  1.  ,
        0.98,  1.  ,  1.  ,  1.  ,  0.99,  1.  ,  1.  ,  0.65,  0.95,
        0.36,  1. ]
WEAK_MEMNET = ["12_baseline", "No Metadata", "WeS Memnet",0.999,  0.572,  0.236,  0.597,  0.837,  0.49 ,  0.639,  0.622,
        0.641,  0.313,  0.7  ,  0.899,  0.803,  0.817,  0.352,  0.495,
        0.491,  0.487,  0.   ,  0.964]
WEAK_MEMNET2 = ["13_baseline", "No Metadata", "PE LS RN JOINT (3-HOP, FB)",1.   ,  0.886,  0.781,  0.866,  0.856,  0.972,  0.817,  0.907,
        0.981,  0.935,  0.997,  0.999,  0.998,  0.931,  1.   ,  0.973,
        0.596,  0.906,  0.12 ,  1.    ]


WEAK_MEMNET_BOW = ["14_baseline", "No Metadata", "BOW 3-Hop",0.994,  0.824,  0.29 ,  0.68 ,  0.817,  0.913,  0.765,  0.886,
        0.789,  0.772,  0.959,  0.997,  0.895,  0.987,  0.757,  0.48 ,
        0.546,  0.519,  0.103,  0.999  ]

WEAK_MEMNET_ONEHOP = ["15_baseline", "No Metadata", " PE LS JOINT (1-HOP, FB)", 0.992,  0.38 ,  0.231,  0.772,  0.89 ,  0.928,  0.841,  0.868,
        0.949,  0.894,  0.916,  0.996,  0.937,  0.631,  0.536,  0.526,
        0.556,  0.904,  0.093,  1. ]



def save_scores(values):
    np.savetxt(experiment_scores_path + values[0]  + ".csv", np.array(values), fmt="%s")

if __name__=="__main__":
    save_scores(task_names_readable)
    save_scores(FB_LSTM_Baseline)
    #save_scores(MEMNET_LSTM_Baseline)
    #save_scores(WEAK_MEMNET)
    save_scores(WEAK_MEMNET2)
    save_scores(WEAK_MEMNET_ONEHOP)

    import glob
    from tabulate import tabulate

    columns = []
    len_first = -1
    for filename in sorted(glob.iglob(experiment_scores_path + "/*.csv")):
        #print filename
        column = []
        try:
            column = np.genfromtxt(filename,dtype='str', delimiter="\t")[2:]
        except:
            print filename, "broken"
            pass
        if(len_first == -1):
            len_first = len(column)
        elif(len_first!=len(column)):
            continue
        
        columns.append(column)

    columns = np.array(columns).T
    c_calc =  np.array(columns.T[1:,1:], dtype = np.float)
    mean = ["Overall Mean"] + list(c_calc.mean(axis = 1))
    columns = np.vstack([columns,mean])
    #if(column[0])

    #print tabulate(columns,headers="firstrow", tablefmt= "latex_booktabs")
    print tabulate(columns,headers="firstrow", tablefmt= "html")








