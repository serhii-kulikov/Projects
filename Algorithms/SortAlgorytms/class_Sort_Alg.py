import numpy.random as rand
class Sort_Alg:
    def Counting_sort(self, min_v, max_v, len_v):
        array = rand.randint(min_v, max_v + 1, len_v)
        m = max_v + 1
        count = [0] * m                
        for a in array:
            count[a] += 1             
        i = 0
        for a in range (m):            
            for c in range(count [a]):  
                array [i] = a
                i += 1
        return array

    def Insertion_sort(self, min_v, max_v, len_v):
        array = rand.randint(min_v, max_v + 1, len_v)
        for i in range (len_v):
            value = array[i]
            index = i - 1
            while index >= 0:
                if value < array [index]:
                    array [index + 1] = array [index]
                    array [index] = value
                    index = index- 1
                else:
                    break
        return array            

    def Heap_sort(self, min_v, max_v, len_v):
        
        def Heapify (array, len_v, i):
            largest = i
            left = 2 * i + 1
            right = 2 * i + 2
            if left < len_v and array [i] < array [left]:
                largest = left
            if right < len_v and array [largest] < array [right]:
                largest = right
            if largest  != i:
                array [i], array [largest] = array [largest], array [i]
                Heapify (array, len_v, largest)
            
        def hSort (array):
            for i in range (len_v // 2, -1, -1):
                Heapify (array, len_v, i)
            for i in range (len_v - 1, 0, -1):
                array [i], array [0] = array [0], array [i]
                Heapify (array, i, 0)
            return array
        
        array = rand.randint(min_v, max_v + 1, len_v)
        return hSort(array)
        
    def Quick_sort(self, min_v, max_v, len_v):

        def partition(array, low, high): 
            i = (low - 1)        
            pivot = array [high]    
            for j in range (low, high): 
                if array [j] <= pivot:  
                    i = i + 1 
                    array [i], array [j] = array [j], array [i] 
            array [i + 1], array [high] = array [high], array [i + 1] 
            return (i + 1)
        
        def qSort (array, low, high): 
            if low < high: 
                pi = partition (array, low, high)  
                qSort (array, low, pi - 1) 
                qSort (array, pi + 1, high)
            return array
        
        array = rand.randint (min_v, max_v, len_v) 
        return qSort (array, 0, len_v - 1)
        
    def Bubble_Sort (self, min_v, max_v, len_v):
        array = rand.randint (min_v, max_v, len_v) 
        for i in range (len_v - 1):
            for j in range(0, len_v - i - 1):
                if array [j] > array [j + 1] :
                    array [j], array [j + 1] = array [j + 1], array [j]
        return array
