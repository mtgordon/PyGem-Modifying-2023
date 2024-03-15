
import FEBio_post_process_driver_as_a_function as ppd

#filepath = "C:\\Users\\Elijah Brown\\Desktop\\Bio Research\\Post Process\\*.feb"

#filepath = "/FeBio Inverse/_Part2_E(1.05)_Part2_v(1.15)_Part9_E(0.80)_Part9_v(1)_Part27_E(0.90)_Part27_v(1).feb"
object_list = ['Object2', 'Object8']
intermediate_csv = '_intermediate.csv'
Results_Folder = 'D:\\Gordon\\Automate FEB Runs\\2023_5_23 auto'
filepath = "C:\\Users\\EGRStudent\\Desktop\\Test_post_process_driver\\*.feb"
ppd.febio_post_function(filepath, object_list, intermediate_csv, Results_Folder)
