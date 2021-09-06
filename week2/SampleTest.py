import sys
import importlib
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--SRN', required=True)

args = parser.parse_args()
subname = args.SRN


try:
   mymodule = importlib.import_module(subname)
except:
    print("Rename your written program as YOUR_SRN.py and run python3.7 SampleTest.py --SRN YOUR_SRN ")
    sys.exit()



def testcase(mymodule):
    cost = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 5, 9, -1, 6, -1, -1, -1, -1, -1],
            [0, -1, 0, 3, -1, -1, 9, -1, -1, -1, -1],
            [0, -1, 2, 0, 1, -1, -1, -1, -1, -1, -1],
            [0, 6, -1, -1, 0, -1, -1, 5, 7, -1, -1],
            [0, -1, -1, -1, 2, 0, -1, -1, -1, 2, -1],
            [0, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1],
            [0, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1],
            [0, -1, -1, -1, -1, 2, -1, -1, 0, -1, 8],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, 0, 7],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0]]
    heuristic = [0, 5, 7, 3, 4, 6, 0, 0, 6, 5, 0]
    
    cost2 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 5, 9, -1, 6, -1, -1, -1, -1, -1, -1],
            [0, -1, 0, 3, -1, -1, 9, -1, -1, -1, -1, -1],
            [0, -1, 2, 0, 1, -1, -1, -1, -1, -1, -1, -1],
            [0, 6, -1, -1, 0, -1, -1, 5, 7, -1, -1, -1],
            [0, -1, -1, -1, 2, 0, -1, -1, -1, 2, -1, -1],
            [0, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1],
            [0, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1],
            [0, -1, -1, -1, -1, 2, -1, -1, 0, -1, 8, -1],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, 0, 7, -1],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0]]
    heuristic2 = [0, 5, 7, 3, 4, 6, 0, 0, 6, 5, 0, 10]
    cost3 = [[0,0,0,0,0,0,0,0,0,0,0],[0,0,6,-1,-1,-1,3,-1,-1,-1,-1],[0,6,0,3,2,-1,-1,-1,-1,-1,-1],[0,-1,3,0,1,5,-1,-1,-1,-1,-1],[0,-1,2,1,0,8,-1,-1,-1,-1,-1],[0,-1,-1,5,8,0,-1,-1,-1,5,5],[0,3,-1,-1,-1,-1,0,1,7,-1,-1],[0,-1,-1,-1,-1,-1,1,0,-1,3,-1],[0,-1,-1,-1,-1,-1,7,-1,0,2,-1],[0,-1,-1,-1,-1,5,-1,3,2,0,3],[0,-1,-1,-1,-1,5,-1,-1,-1,3,0]]
    heuristic3=[0,10,8,5,7,3,6,5,3,1,0]
    start = 1
    goals = [6, 7, 10]
    
    testcases = [
            {
                "A*_testcase": (cost, heuristic, start, goals), # Original Testcase
                "A*_sol": [1,5,4,7],
                "DFS_testcase": (cost,start, goals),
                "DFS_sol": [1, 2, 3, 4, 7]
            },
            {
                "A*_testcase": ([],[],start, goals), # Empty graph
                "A*_sol": [],
                "DFS_testcase": ([],start, goals),
                "DFS_sol": []
            },
            {
                "A*_testcase": (cost, heuristic, 15, goals), # Start point not in graph
                "A*_sol": [],
                "DFS_testcase": (cost, 15, goals),
                "DFS_sol": []
            },
            {
                "A*_testcase": (cost, heuristic, start, []), # Empty goals
                "A*_sol": [],
                "DFS_testcase": (cost, start, []),
                "DFS_sol": []
            },
            {
                "A*_testcase": (cost2, heuristic2, start, [11]), # Goal unreachable
                "A*_sol": [],
                "DFS_testcase": (cost2, start, [11]),
                "DFS_sol": []
            },
            {
                "A*_testcase": (cost2, heuristic2, start, [start]), # Goal same as start state
                "A*_sol": [start],
                "DFS_testcase": (cost2, start, [start]),
                "DFS_sol": [start]
            },
            {
                "A*_testcase": (cost3, heuristic3, start, [10]), # Goal same as start state
                "A*_sol": [1,6,7,9,10],
                "DFS_testcase": (cost3, start, [10]),
                "DFS_sol": [1, 2, 3, 4, 5, 9, 10]
            }
    ]
    
    for index, testcase in enumerate(testcases):
        try:
            if mymodule.A_star_Traversal(*testcase["A*_testcase"])==testcase["A*_sol"]:
                print(f"Test Case {index} for  A* Traversal \033[92mPASSED\033[0m")
            else:
                print("Test Case 1 for  A* Traversal \033[91mFAILED\033[0m")
        except Exception as e:
            print(f"Test Case {index} for  A* Traversal \033[91mFAILED\033[0m due to ", e)


        try:
            if mymodule.DFS_Traversal(*testcase["DFS_testcase"])==testcase["DFS_sol"]:
                print(f"Test Case {index} for DFS Traversal \033[92mPASSED\033[0m")
            else:
                print(f"Test Case {index} for DFS Traversal \033[91mFAILED\033[0m")
        except Exception as e:
            print(f"Test Case {index} for DFS Traversal \033[91mFAILED\033[0m due to ", e)

testcase(mymodule)
