import numpy as np
import os
import open3d as o3d
from sklearn.cluster import DBSCAN
import pandas as pd
from scipy.optimize import curve_fit

def Getmedianpoints(array):
    """
     计算三维矩阵中位点 Calculate the sites in the three-dimensional matrix
    :param array: 三维矩阵 Three-dimensional matrix
    :return: 中位点 median point
    """
    x_arr = np.sort(array[:, 0])
    y_arr = np.sort(array[:, 1])
    z_arr = np.sort(array[:, 2])

    if np.shape(array)[0] % 2 == 0:
        x_median = (x_arr[np.shape(x_arr)[0] // 2] + x_arr[np.shape(x_arr)[0] // 2 - 1]) / 2
        y_median = (y_arr[np.shape(y_arr)[0] // 2] + y_arr[np.shape(y_arr)[0] // 2 - 1]) / 2
        z_median = (z_arr[np.shape(z_arr)[0] // 2] + z_arr[np.shape(z_arr)[0] // 2 - 1]) / 2
        medianpoint = np.array([x_median, y_median, z_median])

    elif np.shape(array)[0] % 2 == 1:
        x_median = x_arr[(np.shape(x_arr)[0] - 1) // 2]
        y_median = y_arr[(np.shape(y_arr)[0] - 1) // 2]
        z_median = z_arr[(np.shape(z_arr)[0] - 1) // 2]
        medianpoint = np.array([x_median, y_median, z_median])

    return medianpoint
#在多个簇中寻找距离最接近的两个簇
def nearest_clu_distance(newarr1):
    """
     在多个簇中寻找距离最接近的两个簇 Look for the two clusters that are closest to each other
    :param newarr1: 经过聚类之后带有标签的四维矩阵 A four - dimensional matrix with labels after clustering
    :return: 距离最接近的两个簇（带标签的四维矩阵） The two closest clusters (labeled 4-dimensional matrix)
    """

    if np.shape(np.unique(newarr1[:, 3]))[0] == 2 :
        return newarr1

    elif np.shape(np.unique(newarr1[:, 3]))[0] >= 3 :
        c = []
        all_distance = []
        for i in np.unique(newarr1[:, 3]):  # 根据特征向量进行关键点的粗过滤
            for j in np.unique(newarr1[:, 3]):
                if i != j:
                    c.append([i, j])
        arr = np.array(c)

        for k in arr:
            index = np.where(newarr1[:, 3] == k[0])
            arr1 = newarr1[index]
            index = np.where(newarr1[:, 3] == k[1])
            arr2 = newarr1[index]
            distance = []
            for p in arr1:
                for o in arr2:
                    d1 = np.sqrt(np.sum(np.square(p[:3] - o[:3])))
                    distance.append(d1)
            all_distance.append([np.array(distance).min(), k[0], k[1]])

        all_distance = np.array(all_distance)
        index = np.where(all_distance[:, 0] == all_distance[:, 0].min())
        need_ij = all_distance[index][0]

        index = np.where(newarr1[:, 3] == need_ij[1])
        arr1 = newarr1[index]
        index = np.where(newarr1[:, 3] == need_ij[2])
        arr2 = newarr1[index]

        return np.vstack((arr1, arr2))

def R2RMSE(pred,merd):    #pred:预测出来的角度；merd:量出来的角度真值,两个一维数组
    """
     计算人工测量角度真值与机器计算角度的相关性 Calculate the correlation between the truth value of manual measurement and the Angle calculated by machine
    :param pred: 计算机计算角度 Computer calculation Angle
    :param merd: 人工测量角度真值 Manual measurement of Angle truth value
    :return: 相关系数和均方根误差 Correlation coefficient and root mean square error
    """
    if np.shape(pred)[0] != 0 :
        popt1, pcov1 = curve_fit(func, merd, pred)
        a1 = popt1[0]  # popt里面是拟合系数
        b1 = popt1[1]
        yfit = func(merd, a1, b1)
        # ---------------R2 and RMSE--------------------------
        SSE = np.sum((pred - yfit) ** 2)  # SSE(和方差、误差平方和)：The sum of squares due to error
        MSE = np.mean((pred - yfit) ** 2)  # MSE(均方差、方差)：Mean squared error
        RMSE = np.sqrt(MSE)
        ymean = np.mean(pred)
        SSR = np.sum((yfit - ymean) ** 2)   # SSR(回归平方和)：The sum of squares total
        SST = np.sum((pred - ymean) ** 2)   # SST(总平方和): The sum of squares regression
        Rsquare = 1.0 * SSR / SST  # (SST-SSE)/SST = 1-SSE/SST
        R = np.sqrt(Rsquare)

    elif np.shape(pred)[0] == 0 :
        print('检测到的点没有真的')
    return R,RMSE

def func(x,a,b):
    return a + b *x #* np.log(x)
# 计算两个簇的最近距离
def twoclu_nearsted_distance(arr1, arr2):
    """
     计算人工测量角度真值与机器计算角度的相关性 Calculate the correlation between the truth value of manual measurement and the Angle calculated by machine
    :param pred: 计算机计算角度 Computer calculation Angle
    :param merd: 人工测量角度真值 Manual measurement of Angle truth value
    :return: 相关系数和均方根误差 Correlation coefficient and root mean square error
    """
    distance = []
    for i in arr1:
        for o in arr2:
            d1 = np.sqrt(np.sum(np.square(i[:3] - o[:3])))
            distance.append(d1)
    return np.array(distance).min()

def cal_angle_of_vector(v0, v1, is_use_deg=True):
    """
     用两个空间向量计算角度 Calculate the Angle with two space vectors
    :param v0: 空间向量 space vector
    :param v1: 空间向量 space vector
    :return: 角度值 Angle value
    """
    dot_product = np.dot(v0, v1)
    v0_len = np.linalg.norm(v0)
    v1_len = np.linalg.norm(v1)
    try:
        angle_rad = np.arccos(dot_product / (v0_len * v1_len))
    except ZeroDivisionError as error:
        raise ZeroDivisionError("{}".format(error))

    if is_use_deg:
        return np.rad2deg(angle_rad)

def sampleReal_branchpoints_angle(file_pathname,infile,excelpath,indexname):
    """
     提取样本人工测量角度值 Take samples and measure the Angle value manually
    :param file_pathname: 存放样本的文件夹 The folder where the samples are stored
    :param infile: 样本名字 Sample name
    :param excelpath: 存放人工测量角度值的excel表 An excel table containing manually measured Angle values
    :param indexname: excel表的索引名 Index name of an excel table
    :return Sample_arr: 人工测量的样本的所有分叉点以及对应的角度值 All bifurcation points and corresponding Angle values of the sample measured manually
    :return df_data: 样本的点云矩阵 Sample point cloud matrix
    """
    file_path = os.path.join(file_pathname, infile)
    #1_将大豆样本划分为数据和标签
    df_data = np.loadtxt(file_path)[:, :3]
    df_lable = np.loadtxt(file_path)[:, 3]
    # 1_标记好大豆单株关键点
    index = np.where(df_lable == 1)
    realkp = df_data[index]
    realkp = DBSCAN_label_without_Noise(realkp)  # **************阈值 0.01*********

    # 2_标记好每个关键点所对应的角度
    # ____________________________________________________________________________________________________________________________________________
    filename = infile[:-4]  # 索引的名字
    real_angel = Files_Real_Angel(excelpath, indexname, filename)
    # _____________________________________________________________________________________________________________________________________________
    medianpoint_arr = []
    # 根据特征向量进行关键点的粗过滤
    for i in np.unique(realkp[:, 3]):
        # 提取出每个聚类，计算簇的中位点
        index = np.where(realkp[:, 3] == i)
        arr = realkp[index]
        if np.shape(arr)[0] > 10:
            arr_median = Getmedianpoints(arr[:, :3])
            medianpoint_arr.append(arr_median)

    medianpoint_arr = np.asarray(medianpoint_arr)
    # np.savetxt('medianpoint_arr.txt',medianpoint_arr)
    sort_idx = np.argsort(medianpoint_arr[:, 2])[::]  # argsort()从小到大排序
    medianpoint_arr = medianpoint_arr[sort_idx]
    Sample_arr = np.column_stack((medianpoint_arr, real_angel.T[::-1] ))

    return Sample_arr,df_data

def Get2medianpoints(newarr):
    """
     获得两个簇的中位点 Obtain the median points of the two clusters
    :param newarr: 两个簇 Two clusters
    :return: 两个中位点 two median points
    """
    y = np.unique(newarr[:, 3])
    index = np.where(newarr[:, 3] == y[0])
    arr4 = newarr[index][:, :3]
    mu0 = Getmedianpoints(arr4)
    index = np.where(newarr[:, 3] == y[1])
    arr5 = newarr[index][:, :3]
    mu1 = Getmedianpoints(arr5)

    return mu0, mu1

def Files_Real_Angel(excelpath, indexname, filename):  # 通过文件名 找到它对应的角度真值
    """
     读取样本角度真值 Read sample Angle truth value
    :param excelpath: 存放人工测量角度值的excel表 An excel table containing manually measured Angle values
    :param indexname: excel表的索引名 Index name of an excel table
    :param filename: infile: 样本名字 Sample name
    :return: 样本角度真值 Sample Angle truth value
    """
    data = pd.read_excel(excelpath,
                         sheet_name="Sheet1")
    data.set_index(indexname, inplace=True)
    real_angel = np.asarray(data.loc[filename])
    # 去除nan值
    n = -1
    index = []
    for i in real_angel:
        n += 1
        if np.isnan(i):
            index.append(n)

    real_angel = np.delete(real_angel, index)

    return real_angel

def Find_extend_clu(medianpoint1,mu):
    """
     寻找分叉点对应的两个分枝簇 Look for two branch clusters corresponding to the bifurcation point
    :param medianpoint1: 存放上层中位点的矩阵 The matrix that holds the upper locus
    :param mu: 下层分叉点 Lower bifurcation point
    :return: 分叉点对应的两个分枝簇 Two branch clusters corresponding to the bifurcation point
    """
    distance = []
    for i in medianpoint1:
        find_min_distance = []
        d1 = np.sqrt(np.sum(np.square(i[:2] - mu[:2])))
        find_min_distance.append(d1)
        distance.append([i[3],np.array(find_min_distance).min()])
    distance = np.array(distance)
    sort_idx = np.argsort(distance[:, 1])[::1]  # argsort()从小到大排序
    distance = distance[sort_idx]
    k0 = distance[0][0]
    k1 = distance[1][0]
    label = np.array([k0,k1])
    return label

def accord_label_get_array(array,label):
    """
     根据label找到对应簇 Locate the corresponding cluster by label
    :param array: 存放上层簇的矩阵 The matrix that holds the upper cluster
    :param label: 标签 label
    :return: 标签对应的两个分枝簇 The label corresponds to two branch clusters
    """
    count = 0
    for i in label :
        count += 1
        index = np.where(array[:,3]==i)
        arr = array[index]
        if count == 1 :
            newarr = arr
        else :
            newarr = np.vstack((newarr,arr))
    return newarr


def get_k_neighbors(points, center, k):
    """
    获取三维点云中以中心点为中心的K领域内的点，并去除多余的点，返回一个三维坐标数组
    :param points: 三维点云，Nx3的数组，表示N个点的三维坐标
    :param center: 三维点，长度为3的数组，表示中心点
    :param k: 搜索半径，int类型
    :return: 三维坐标数组，表示K领域内的点
    """
    distances = np.linalg.norm(points - center, axis=1)
    #将points和distances进行重排序
    points = points[np.argsort(distances)]
    distances = distances[np.argsort(distances)] # 计算每个点到中心点的距离
    neighbors_indices = np.where(distances <= k)[0]  # 找到所有符合条件的点在原数据中的索引
    if len(neighbors_indices) == 0:
        index = 0
        # 如果找不到中心点领域内的点，则返回(0, 0, 0)
        return np.array([0, 0, 0]),index,points
    else:
        # 如果找到中心点领域内的点，则去除其中的重复点并返回
        unique_neighbors = points[neighbors_indices[0]]#选出最近的点
        index = len(neighbors_indices)
        points = np.delete(points,neighbors_indices,axis=0)
        return unique_neighbors,index,points

def Keypoint_PRFestamate(branch_point,Sample_arr,threshold = 0.01) : #两个矩阵都是四行n列，其中前三列是中位点的XYZ，第四列是每个点对应的角度
    """
     分叉点定位精度验证
    :param branch_point: 机器寻找的分叉点
    :param Sample_arr: 人工标注的分叉点
    :param threshold: 机器寻找分叉点与人工标注分叉点的距离，当距离小于threshold时，即为寻找正确
    :return: TP,FP,FN,TRUE_branch_ponit TP表示正检，FP表示过检，FN表示漏检，TRUE_branch_ponit表示正确寻找的点
    """
    Sample_num = np.shape(Sample_arr)[0] #样本中的信息条数
    Predicted_arr = branch_point
    Sample_arr = Sample_arr[:,:3]
    Predicted_num = np.shape(Predicted_arr)[0] #提取出的信息条数


    True_Predicted_arr = np.zeros(np.shape(Sample_arr)) #创建存取正确关键点的零矩阵
    FP,FN,TP = 0,0,0
    if Predicted_num == 0 :
        FN += Sample_num
    else:
        count = -1  # 这是用作定位的，定位到目前的点和角度
        for i in Sample_arr:
            unique_neighbors,index_num,Predicted_arr = get_k_neighbors(Predicted_arr, i, threshold)
            Predicted_num = Predicted_num - index_num
            count += 1
            if unique_neighbors[0] == 0 :
                FN += 1
            else :
                True_Predicted_arr[count] = unique_neighbors
                TP += 1
        FP = Predicted_num

    TRUE_branch_ponit  = np.array(True_Predicted_arr) #要把正确的矩阵拿出来
    return TP,FP,FN,TRUE_branch_ponit #返回的是跟样本矩阵一样形状的正确点的矩阵

def DBSCAN_label_without_Noise(array, k=0.005):  # DBSCAN聚类且没有噪声，返回的是携带标签的矩阵
    """
      用DBSCAN进行聚类并去除噪声标签 Cluster and remove noise label with DBSCAN
     :param array: 原矩阵
     :return: 进行完聚类的矩阵
     """
    clustering = DBSCAN(eps=k).fit(array)  # **************阈值*********
    label = clustering.labels_.T
    newarr1 = np.column_stack((array, label))
    index = np.where(newarr1[:, 3] == -1)
    newarr1 = np.delete(newarr1, index, axis=0)
    return newarr1

def DBSAN_label_with_pointnumber(array, filter_number=5):  # 将DBSCAN后的矩阵 点数小于filter_number的去除
    """
      用DBSCAN进行聚类并去除噪声点 Cluster and remove noise points with DBSCAN
     :param array: 原矩阵
     :return: 进行完聚类并去除噪声点的矩阵
     """

    for i in np.unique(array[:, 3]):
        # 提取出每个聚类，计算簇的中位点
        index = np.where(array[:, 3] == i)
        arr = array[index]
        if np.shape(arr)[0] <= filter_number:
            array = np.delete(array, index, axis=0)
        else:
            pass
    return array
# 上下相同聚类层判断
def issame_nearby_slice(arr1, arr2):  # 聚类中位点,阈值越小找到的越多
    """
      上下相同聚类层判断
     :param arr1: 上矩阵
     :param arr2: 下矩阵
     :return: 判断有无新分枝
     """

    all_distance = []
    for i in arr1:
        distance = []
        for o in arr2:
            d1 = np.sqrt(np.sum(np.square(i[:3] - o[:3])))
            distance.append(d1)
        all_distance.append(np.array(distance).min())
    all_distance = np.array(all_distance)

    if all_distance.max() > 2*all_distance.min():
        return False
    else:
        return True
# 上小下大聚类层判断
def isnew_slice(arr1, arr2):  # 聚类中位点，阈值越小找的的越多
    """
      上小下大聚类层判断
     :param arr1: 上矩阵
     :param arr2: 下矩阵
     :return: 判断有无新分枝
     """

    count = 0
    all_distance = []
    for i in arr1:
        distance = []
        for o in arr2:
            d1 = np.sqrt(np.sum(np.square(i[:3] - o[:3])))
            distance.append(d1)
        all_distance.append(np.array(distance).min())

    all_distance_arr = np.array(all_distance)
    for i in all_distance :
        if i < 2*all_distance_arr.min() :
            count += 1
        else:
            continue

    if count >= np.shape(arr2)[0]:
        return False
    else:
        return True
#其他的都是跟找到分枝有关，下面这个函数用于找到分枝点。
def Growdown_find_branchpoints(right_array, df_data, cricle_num=100,knn_num = 40):
    """
      向下区域增长寻找分叉点 Look for bifurcation points in downward regional growth
     :param right_array: 寻找到的新分枝存在的层 Find the layer where new branches exist
     :param df_data: 原数据 Raw data
     :param cricle_num: 循环计算次数 Cycle count
     :param knn_num: 向下区域生长点 Downward growth point
     :return: 新的可能分叉点 new bifurcation points
     """

    # 向下寻找关键点
    # 1_先将他聚成两类
    #     right_array = DBSCAN_label_without_Noise(right_array,k=eps)

    y = np.unique(right_array[:, 3])
    index = np.where(right_array[:, 3] == y[0])
    arr0 = right_array[index][:, :3]
    index = np.where(arr0[:, 2] == arr0[:, 2].min())
    point0 = arr0[index][0]

    index = np.where(right_array[:, 3] == y[1])
    arr1 = right_array[index][:, :3]
    index = np.where(arr1[:, 2] == arr1[:, 2].min())
    point1 = arr1[index][0]
    # 2_截取出一定的区域 减少数据量
    up_floor = np.array([point0[2], point1[2]]).max()
    arrz = df_data[:, 2]
    index = np.where((arrz >= up_floor - 0.05) & (arrz < up_floor))
    testarr = df_data[index]
    # 3_kdtree近邻搜索，搜索出的
    # 将矩阵用o3d显示出来
    pcd = o3d.geometry.PointCloud()  # 首先建立一个pcd类型的数据（这是open3d中的数据形式）
    testarr = np.vstack((testarr, point0, point1))

    list = [1]
    for q in list:
        list.append(q)  # 无限循环
        if len(list) > cricle_num:
            key_points = []
            break

        pcd.points = o3d.utility.Vector3dVector(testarr)  # 将点云转换成open3d中的数据形式并用pcd来保存，以方便用open3d处理
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)

        # ——————K近邻搜索————————
        [k1, idx1, _] = pcd_tree.search_knn_vector_3d(pcd.points[len(testarr) - 2], knn_num)
        find_points1 = np.asarray(pcd.points)[idx1[1:], :]

        arr0 = np.unique(np.vstack((arr0, find_points1)), axis=0)
        #         index = np.where(find_points1[:,2]==find_points1[:,2].min())
        #         point0 = np.squeeze(find_points1[index])
        point0 = Getmedianpoints(find_points1)
        # ——————半径搜索————————
        #         [k1, idx1, _] = pcd_tree.search_radius_vector_3d(pcd.points[len(testarr)-2], 0.005)
        #         find_points1 = np.asarray(pcd.points)[idx1[1:], :]
        #         arr0 = np.unique(np.vstack((arr0,find_points1)),axis=0)

        [k2, idx2, _] = pcd_tree.search_knn_vector_3d(pcd.points[len(testarr) - 1], knn_num)
        find_points2 = np.asarray(pcd.points)[idx2[1:], :]

        arr1 = np.unique(np.vstack((arr1, find_points2)), axis=0)
        #         index = np.where(find_points2[:,2]==find_points2[:,2].min())
        #         point1 = np.squeeze(find_points2[index])
        point1 = Getmedianpoints(find_points2)

        up_floor = np.array([point0[2], point1[2]]).max()
        index = np.where(testarr[:, 2] <= up_floor)
        testarr = testarr[index]

        # 判断两个簇里面相同的行
        count_num = 0
        key_points = []
        for i in range(len(arr0)):  # generate pairs
            for j in range(len(arr1)):
                if np.array_equal(arr0[i], arr1[j]):
                    count_num += 1
                    key_points.append(arr0[i])
                else:
                    continue

        if count_num >= knn_num:
            key_points = np.array(key_points)
            break
        else:
            testarr = np.vstack((testarr, point0, point1))

    return key_points

#本实验的主要函数，分叉点定位
def Bifurcation_point_location(df_data,H = 0.01,N = 80): #计算样本的关键点及角度,value：分层再往上找VALUE层作为分枝点
    """
      分枝点寻找 Branch detection
     :param df_data: 原数据 Raw data
     :param H: 分层高度 Slice height
     :param N: 向下区域增长点数 Downward zone growth point
     :return: 新的分叉点 New bifurcation points
     """
    # 1_大豆点云离散点去除
    L = H
    knn_num = N

    # df_data = remove_radius_noise(df_data)

    # 2_大豆单株点云摆直
    # ****************************************************************************************************

    # 2_切片按距离L从Z轴从下往上切片
    height = df_data[:, 2].max() - df_data[:, 2].min()  # 求出高度
    gapnumber = int(height // L + 1)
    arrz = df_data[:, 2]
    Slice_growing = []
    n = -1
    # 创造字典
    for i in range(0, gapnumber):
        index = np.where((arrz >= arrz.min() + i * L) & (arrz[:] < arrz.min() + (i + 1) * L))
        slice_arr = df_data[index]  # 切片出来
        if np.shape(slice_arr)[0] > 10:
            slice_arr = DBSCAN_label_without_Noise(slice_arr)  # 将切片层进行聚类
            slice_arr = DBSAN_label_with_pointnumber(slice_arr)
            clu_num = np.shape(np.unique(slice_arr[:, 3]))[0]  # 聚类数量
            n += 1
            count = 0
            for k in np.unique(slice_arr[:, 3]):
                count += 1
                # 提取出每个聚类，计算簇的中位点
                index = np.where(slice_arr[:, 3] == k)
                karr = slice_arr[index][:, :3]
                kmedianpoint = np.hstack((Getmedianpoints(karr), k))
                if count == 1:
                    medianpoint_arr = np.expand_dims(kmedianpoint, axis=0)
                else:
                    medianpoint_arr = np.vstack((medianpoint_arr, kmedianpoint))

            Slice_growing.append(
                {'序号': n, '聚类数量': clu_num, '高度': (i + 1) * L, '聚类中位点': medianpoint_arr, '矩阵': slice_arr})

    clusternumber = []
    height = []
    for i in range(0,len(Slice_growing)) :
       clusternumber.append(Slice_growing[i]['聚类数量'])
       height.append(Slice_growing[i]['高度'])
    y = np.array(clusternumber)
    x = np.array(height)

    # 3_分枝检测
    possible_point = []
    for i in range(1, len(x) -1):
        y0_value = y[i + 1]
        y1_value = y[i]
        y2_value = y[i - 1]
        if y1_value > 1 :
            if y1_value == y2_value:
                if not issame_nearby_slice(Slice_growing[i-1]['聚类中位点'], Slice_growing[i]['聚类中位点']):
                    possible_point.append(x[i])
            elif y1_value - y2_value > 0:
                if y0_value == y2_value:
                    continue
                else:
                    possible_point.append(x[i])
            elif y1_value - y2_value < 0:
                if isnew_slice(Slice_growing[i-1]['聚类中位点'], Slice_growing[i]['聚类中位点']):
                    possible_point.append(x[i])
    possible_point = np.array(possible_point)

    # 4_分叉点定位
    branch_point = []
    for i in possible_point:
        for m in range(0, len(Slice_growing)):
            if Slice_growing[m]['高度'] == i:
                find_key_points = nearest_clu_distance(Slice_growing[m]['矩阵'])
                # mu_o, mu_t = Get2medianpoints(find_key_points)
                # medianpoint2 = np.array([mu_o, mu_t])
                key_points = Growdown_find_branchpoints(find_key_points, df_data,knn_num = knn_num)
                if len(key_points) > 0:
                    mu = Getmedianpoints(key_points)
                    branch_point.append(mu)
                elif len(key_points) == 0:
                    continue
            else:
                continue
    branch_point = np.array(branch_point)
    return branch_point

def Branch_point_optimization(df_data, TRUE_branch_ponit, D=0.023):
    """
      分枝点优化 Branch optimization
     :param df_data: 原数据 Raw data
     :param TRUE_branch_ponit: 寻找到的分叉点矩阵 The bifurcation matrix is found
     :param D: 向上优化距离 Upward optimization distance
     :return: 分枝角度 branch angle
     """
    arrz = df_data[:, 2]
    cut_radius = D
    total_angle_vector = []
    for mu in TRUE_branch_ponit:
        # 将关键点上方0.015-0.03的地方切出来来计算
        if mu[2] == 0 :
            angle_vector = 0
            total_angle_vector.append(angle_vector)
            continue
        index = np.where((arrz >= mu[2] + cut_radius) & (arrz[:] < mu[2] + cut_radius + 0.01))
        cut_arr = df_data[index]
        if np.shape(cut_arr)[0] == 0:
            angle_vector = 0
            total_angle_vector.append(angle_vector)
            continue
        cut_arr = DBSCAN_label_without_Noise(cut_arr)  # 将切片层进行聚类
        cut_arr = DBSAN_label_with_pointnumber(cut_arr)
        # 计算出切片层的中位点
        mediancount = 0
        for k in np.unique(cut_arr[:, 3]):
            mediancount += 1
            # 提取出每个聚类，计算簇的中位点
            index = np.where(cut_arr[:, 3] == k)
            karr = cut_arr[index][:, :3]
            kmedianpoint = np.hstack((Getmedianpoints(karr), k))
            if mediancount == 1:
                medianpoint1 = np.expand_dims(kmedianpoint, axis=0)
            else:
                medianpoint1 = np.vstack((medianpoint1, kmedianpoint))

        if mediancount >= 2:
            label = Find_extend_clu(medianpoint1, mu)
            right_array = accord_label_get_array(cut_arr, label)
            mu0, mu1 = Get2medianpoints(right_array)
            angle_vector = cal_angle_of_vector((mu - mu0), (mu - mu1), is_use_deg=True)
            # if angle_vector >= 90:
            #     angle_vector = 180 - angle_vector
            total_angle_vector.append(angle_vector)
        else:
            angle_vector = 0
            total_angle_vector.append(angle_vector)

    total_angle_vector = np.array(total_angle_vector)
    return total_angle_vector

if __name__ == '__main__':
    work_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))#../
    file_pathname = '%s\\dataset\\Soybean3D'%work_dir #放数据集的位置test
    #存放真值的excel文件夹
    excelpath = "%s\\dataset\\Simples_GT.xlsx"%work_dir  # 要打开的excel文件
    indexname = '文件名'  # 要以哪一个名字为索引
    #参数
    H_list = [0.01]#,0.005,,0.015,0.02,0.025,0.03
    N_list = [60]#20,40,,80,100,120,140
    D_list = [0.03]# 0.015, 0.02, 0.025,,0.035,0.04
    dic_p,dic_r,dic_F,dic_R = {'N\H':N_list},{'N\H':N_list},{'N\H':N_list},{'D':D_list}
    for H in H_list:
        P_list,R_list,F_list = [],[],[]
        for N in N_list :
            all_branchpoint = []
            cc = 0  # 记载每个文件的序号
            num = 0
            predict_num = 0
            for infile in os.listdir(file_pathname):
                cc += 1
                print('H:%s_N:%s'%(H,N)+'__%s_Done!'%cc)
                # 1_正确样本构建
                Sample_arr,df_data = sampleReal_branchpoints_angle(file_pathname, infile, excelpath, indexname) #file_pathname, infile:用来指示文件 ； excelpath, indexname：用来指示excel
                num += np.shape(Sample_arr)[0]

                # 2_计算出每个样本的关键点并进行精度评价
                branch_ponit = Bifurcation_point_location(df_data, H=H, N=N)
                predict_num += np.shape(branch_ponit)[0]

                TP_l,FP_l,FN_l,TRUE_branch_ponit = Keypoint_PRFestamate(branch_ponit,Sample_arr,threshold = 0.01)
                all_branchpoint.append(TRUE_branch_ponit)

                if cc == 1:
                    TP = TP_l
                    FP = FP_l
                    FN = FN_l
                else :
                    TP = TP_l+TP
                    FP = FP_l+FP
                    FN = FN_l+FN

            print('*****************(H：%s; N:%s)accuracy of branch detection：*****************'%(H,N))
            Precision = TP / (TP + FP)
            print('  Precision:', '%.2f' % (Precision * 100), '%')

            Recall = TP / (TP + FN)
            print('  Recall:', '%.2f' % (Recall * 100), '%')

            F_score = Precision * Recall * 2 / (Precision + Recall)
            print('  F-score:', '%.2f' % F_score)

            P_list.append(Precision)
            R_list.append(Recall)
            F_list.append(F_score)

            r_list = []
            for D in D_list:
                count = -1
                angle_num = 0
                for infile in os.listdir(file_pathname):
                    count += 1
                    # print(infile)
                    Sample_arr,df_data = sampleReal_branchpoints_angle(file_pathname, infile, excelpath, indexname)
                    TRUE_branch_ponit = all_branchpoint[count]
                    # 3_通过正确的关键点计算角度
                    total_angle_vector = Branch_point_optimization(df_data, TRUE_branch_ponit, D=D)
                    angle_num += np.shape(Sample_arr[:,3])[0]
                    # print(total_angle_vector)
                    # print(Sample_arr[:,3])
                    #4_计算模型的关键节点F值以及角度R值
                    if count == 0:
                        all_real_angle = Sample_arr[:,3]
                        all_angle_arr = total_angle_vector
                    else :
                        all_real_angle = np.append(all_real_angle,Sample_arr[:,3])
                        all_angle_arr = np.append(all_angle_arr,total_angle_vector)
                index = np.where((all_angle_arr == 0))
                all_real_angle = np.delete(all_real_angle, index)
                all_angle_arr = np.delete(all_angle_arr, index)

                # # 输出到excel中,角度对比
                # dic = {'Measured angle': all_angle_arr,
                #        'Predicted angle': all_real_angle}
                # df = pd.DataFrame(dic)
                # df.to_csv('%s\\results\\SCAG_best_Fsocre.csv'%work_dir)
                #
                R,RMSE = R2RMSE(all_real_angle, all_angle_arr)
                r_list.append(R)
                print('(D：%s)角度计算r值：'%D,R)

    # 输出到excel中,做参数敏感性分析
            dic_R['H:%s/N:%s' %(H,N)] = r_list
            df_R = pd.DataFrame(dic_R)
        dic_p['%s' % H],dic_r['%s' % H],dic_F['%s' % H] = P_list,R_list,F_list

        df_p,df_r,df_F = pd.DataFrame(dic_p),pd.DataFrame(dic_r),pd.DataFrame(dic_F) #缩进了一tab
        with pd.ExcelWriter('%s\\results\\SCAG_F_sensitivity.xlsx'%work_dir) as writer:
            df_p.to_excel(writer,sheet_name='Precision',index=False)
            df_r.to_excel(writer,sheet_name='Recall',index=False)
            df_F.to_excel(writer,sheet_name='F-score',index=False)
            df_R.to_excel(writer, sheet_name='r_value', index=False)