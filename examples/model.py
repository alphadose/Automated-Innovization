import numpy as np
import pandas as pd
import time, math
#import matplotlib.pyplot as plt
from itertools import combinations, product

class AutomatedInnovization:
    
    def __init__(self, dataset=None, basis_functions=None, drop_duplicates=False, plot_c_values=False):
        self.data = np.log(dataset[basis_functions]).groupby(basis_functions).size().reset_index()
        self.basis_functions = list(self.data[basis_functions].nunique().sort_values(ascending=False).index)
        self.data = self.data[[0] + self.basis_functions]
        self.data.columns = ["Frequency"] + self.basis_functions
        if drop_duplicates:
            self.data["Frequency"] = 1
        self.frequencies = self.data["Frequency"].values
        self.cluster_threshold = 3
        self.plot_c_values = plot_c_values
        self.index = self.basis_functions.index(basis_functions[0])
    
    def __prettyPrint(self, gradient, total_points, counter, constants, rule_index):
        total_clusters = sum(counter >= self.cluster_threshold)
        total_clustered_points = sum(counter[counter >= self.cluster_threshold])
        
        print("Rule {}:-\n".format(rule_index))
            
        print("Equation:   {} = C\n".format(" * ".join(map("{}^({})".format, self.basis_functions, gradient))))
            
        print("Total Significance:   {}% ({} points out of {})\n"
              .format(round(total_clustered_points*100/total_points, 2), total_clustered_points, total_points))
          
        clustered_c_index, clustered_c, unclustered_c_index, unclustered_c = [], [], [], []
        loop_index, cluster_index = 1, 1
          
        for constant, count in zip(constants, counter):
            
            if count >= self.cluster_threshold:
                print("Cluster {}/{}:-".format(cluster_index, total_clusters))
                
                cluster_index += 1

                print("C:   {}".format(constant))

                print("Significance:   {}% ({} points out of {})\n".format(count*100/total_points, count, total_points))

                for loop_index in range(loop_index, loop_index + count):
                    clustered_c.append(constant)
                    clustered_c_index.append(loop_index)
            else:
                for loop_index in range(loop_index, loop_index + count):
                    unclustered_c.append(constant)
                    unclustered_c_index.append(loop_index)
            loop_index += 1
        
        if self.plot_c_values:  
            plt.scatter(clustered_c_index, clustered_c, 10, color='k', label='Clustered Point')
            plt.scatter(unclustered_c_index, unclustered_c, 20, facecolors='none', edgecolors='k', label='Unclustered Point')
            plt.locator_params(integer=True)
            plt.xlabel('Data Points', fontsize=14)
            plt.ylabel('c Values', fontsize=14)
            plt.tight_layout()
            plt.legend()
            plt.show()
        
    
    def __getGradients(self):
        gradient_list = [[] for i in self.basis_functions[1:]]
        counts = []
        
        for i, j in combinations(self.data.values, 2):
            x1, x2 = i[1], j[1]
            counts.append(i[0]*j[0])
            
            for index, (y1, y2) in enumerate(zip(i[2:], j[2:])):
                if y1 == y2:
                    gradient_list[index].append(0)
                elif x1 == x2:
                    gradient_list[index].append(90)
                else:
                    gradient_list[index].append(math.degrees(math.atan((y2 - y1)/(x2 - x1))))
        
        return np.array(gradient_list), np.array(counts).astype(int)
    
    
    def __getConstants(self, gradient):
        c = np.repeat((self.data[self.basis_functions].values * gradient).sum(axis = 1), 3)
        counts = np.repeat(self.frequencies, 3)
        
        for index, (constant, count) in enumerate(sorted(zip(c, counts))):
            c[index] = constant
            counts[index] = count
        
        total_points = sum(counts)
        dc = np.gradient(c)
        std = np.std(dc)
        
        threshold = int(math.sqrt(len(c)))
        leftover = len(c) - threshold**2
        
        clusters = [0]
        
        cluster_start_index = 0
        
        for i in range(threshold):
            cluster_end_index = cluster_start_index + threshold + 1 if i < leftover else cluster_start_index + threshold
            std_coef = np.std(dc[cluster_start_index:cluster_end_index])
            
            if math.sqrt(3)*std_coef < std:
                std_coef = std
            upper_limit = math.sqrt(3)*std_coef
            
            for j in range(cluster_start_index, cluster_end_index):
                if dc[j] > upper_limit and j%3 == 0:
                    clusters.append(j)
            cluster_start_index = cluster_end_index
        
        clusters.append(len(c))
        
        constants = []
        counter = []
        for i in range(len(clusters) - 1):
            start, end = clusters[i], clusters[i+1]
            if start != end+1:
                constants.append(math.e**(sum(c[start:end]*counts[start:end])/sum(counts[start:end])))
                counter.append(sum(counts[start:end])//3)
        
        return total_points//3, np.array(counter).astype(int), np.array(constants)
    
    
    def __getLimits(self, buckets):
        limit_list = []
        minimum_limit_threshold = sum(self.frequencies)
        
        for bucket in buckets:
            safe_bucket = [0, 0] + bucket + [0, 0]
            limits = [-2]

            for i in range(0, len(bucket) - 1):
                above_limit_threshold = i > limits[-1]+1 and max(safe_bucket[i+2], safe_bucket[i+3]) > minimum_limit_threshold
                is_local_maxima = safe_bucket[i+2]+safe_bucket[i+3] > max(safe_bucket[i]+safe_bucket[i+1], safe_bucket[i+4]+safe_bucket[i+5])
                
                if above_limit_threshold and is_local_maxima:
                    limits.append(i)
            
            limit_list.append(limits[1:])
            
        return limit_list

    
    def findRules(self):
        start_time = time.time()
        
        result_index = self.basis_functions
        power_list = [[1]]
        
        gradient_list, counts = self.__getGradients()
        
        maximas, minimas = [max(i) for i in gradient_list], [min(i) for i in gradient_list]
        bucket_size = int(math.sqrt(len(gradient_list[0])))
        
        buckets = [[0 for i in range(bucket_size)] for j in range(len(self.basis_functions) - 1)]
        
        for bucket_index, (gradients, maxima, minima) in enumerate(zip(gradient_list, maximas, minimas)):
            for i, j in zip(gradients, counts):
                index = min(int(((i-minima)*bucket_size)/(maxima-minima)), bucket_size-1)
                buckets[bucket_index][index] += j
        
        limit_list = self.__getLimits(buckets)
        
        for limits, gradients, maxima, minima, x in zip(limit_list, gradient_list, maximas, minimas, self.basis_functions[1:]):
            
            powers = []
            
            for limit in limits:
                upper_limit, lower_limit = minima + (maxima-minima)*(limit+2)/bucket_size, minima + (maxima-minima)*(limit)/bucket_size
                bucket_gradients = gradients[(gradients <= upper_limit) & (gradients >= lower_limit)]
                bucket_counts = counts[(gradients <= upper_limit) & (gradients >= lower_limit)]

                while True:

                    bucket_maximum, bucket_minimum = max(bucket_gradients), min(bucket_gradients)

                    threshold = int(math.sqrt(len(bucket_gradients)))

                    if threshold <= 3 or bucket_minimum == bucket_maximum:
                        current_gradient = sum(bucket_gradients*bucket_counts)/sum(bucket_counts)
                        break

                    buckets = [0 for i in range(threshold)]

                    for bg, bc in zip(bucket_gradients, bucket_counts):
                        index = min(int(((bg - bucket_minimum)*threshold)/(bucket_maximum - bucket_minimum)), threshold-1)
                        buckets[index] += bc
                    
                    cap_index = buckets.index(max(buckets))
                    
                    upper_limit = bucket_minimum + (bucket_maximum - bucket_minimum)*(cap_index+2)/threshold
                    lower_limit = bucket_minimum + (bucket_maximum - bucket_minimum)*(cap_index-1)/threshold
                    
                    bucket_counts = bucket_counts[(bucket_gradients <= upper_limit) & (bucket_gradients >= lower_limit)]
                    bucket_gradients = bucket_gradients[(bucket_gradients <= upper_limit) & (bucket_gradients >= lower_limit)]
                
                if current_gradient != 0:
                    powers.append(1 / np.float64(math.tan(- current_gradient * math.pi / 180)))
            
            power_list.append(powers)
        
        rules = []

        power_list = list(product(*power_list))

        for powers in power_list:
            rule = np.array(list(powers))/powers[self.index]
            rules.append((rule, self.__getConstants(rule)))
        
        print("################################################################################################\n")
        print(" vs ".join(self.basis_functions)+"\n")
        print("Time taken: {} seconds\n".format(time.time() - start_time))
            
        for rule_index, (gradient, (total_points, counter, constants)) in enumerate(rules):
            self.__prettyPrint(gradient, total_points, counter, constants, rule_index + 1)
            
        print("################################################################################################\n")
