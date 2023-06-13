import numpy as np
import os, math
from sklearn.cluster import DBSCAN
import open3d as o3d
from sklearn.cluster import KMeans
import pandas as pd
from scipy.optimize import curve_fit
from scipy import spatial

def keypoint_detact(array,k,r):  # 基于球形的关键点提取
    """
     关键簇检测 Critical cluster detection
    :param array: 点云矩阵 Point cloud matrix
    :param k: 提取出前百分之K的点 take the top K percent
    :param r: 每个点的搜索半径 Search radius for each point
    :return: 可能存在分叉点的簇矩阵 A cluster matrix with possible bifurcation points
    """
    count = []
    tree = spatial.KDTree(array)
    for point in array :
        results = tree.query_ball_point(point, r)
        nearby_points = array[results]
        n = np.shape(nearby_points)[0]
        count.append(n)
    count_arr = np.array(count)
    new_arr = array[np.argsort(-count_arr)]
    new_arr = new_arr[:k, :]  # *********提取点数*********
    return new_arr

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

def Keypoint_PRFestamate(Predicted_arr,Sample_arr) : #两个矩阵都是四行n列，其中前三列是中位点的XYZ，第四列是每个点对应的角度
    """
     分叉点定位精度验证
    :param branch_point: 机器寻找的分叉点
    :param Sample_arr: 人工标注的分叉点
    :param threshold: 机器寻找分叉点与人工标注分叉点的距离，当距离小于threshold时，即为寻找正确
    :return: TP,FP,FN,TRUE_branch_ponit TP表示正检，FP表示漏检，FN表示过检，TRUE_branch_ponit表示正确寻找的点
    """
    Sample_num = np.shape(Sample_arr)[0] #样本中的信息条数
    Predicted_num = np.shape(Predicted_arr)[0] #提取出的信息条数

    n = 0 #这是用作计算正确数量的
    count = -1 #这是用作定位的，定位到目前的点和角度
    True_Predicted_arr = np.zeros(np.shape(Sample_arr)) #创建存取正确关键点的零矩阵
    for i in Predicted_arr[:,:3] :
        go = 0
        count += 1
        count2 = -1  # 这是用作定位的，定位到样本的点和角度
        for j in Sample_arr[:,:3] :
            count2 += 1
            if go == 0:
                d1 = np.sqrt(np.sum(np.square(i - j)))
                if d1 <= 0.012 : #*****************************************************
                    True_Predicted_arr[count2] =Predicted_arr[count]
                    go += 1
                    n +=1
                else:
                    pass
            else :
                continue
    Extract_true = n #提取出正确的条数
    True_Predicted_arr = np.asarray(True_Predicted_arr) #要把正确的矩阵拿出来
    return Extract_true,Sample_num,Predicted_num,True_Predicted_arr #返回的是跟样本矩阵一样形状的正确点的矩阵

def DBSCAN_label_without_Noise(array,k=0.005): #DBSCAN聚类且没有噪声，返回的是携带标签的矩阵
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

def func(x,a,b):
    return a + b *x #* np.log(x)

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
        SSR = np.sum((yfit - ymean) ** 2)
        SST = np.sum((pred - ymean) ** 2)
        Rsquare = 1.0 * SSR / SST  # (SST-SSE)/SST = 1-SSE/SST
        R = np.sqrt(Rsquare)

    elif np.shape(pred)[0] == 0 :
        print('检测到的点没有真的')
    return R,RMSE

def Files_Real_Angel(excelpath,indexname,filename): #通过文件名 找到它对应的角度真值
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

    return  real_angel

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
        unique_neighbors = points[neighbors_indices[0]]#unique_neighbors, _ = np.unique(points[neighbors_indices], axis=0, return_index=True)
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

def PCA_delete_wrongpoints(originaldf,newarr1):
    """
     使用PCA的方法去除不包含分叉点的关键簇 The key clusters without bifurcation points were removed by PCA
    :param originaldf: 原始点云 Primary point cloud
    :param newarr1: 寻找出来的可能存在分叉点的簇 Look for clusters that may have bifurcation points
    :return: 去除完不包含分叉点簇后的关键簇 After removing the key cluster that does not contain the bifurcated point cluster
    """
    # 判断提取的关键点正确性,参数：y，z 副本 根据区域增长进行判断 过滤：特征向量
    l = 0
    m = np.array([])
    # 根据特征向量进行关键点的粗过滤
    for i in np.unique(newarr1[:, 3]):
        # 提取出每个聚类，计算簇的中位点
        index = np.where(newarr1[:, 3] == i)
        arr = newarr1[index]
        # if np.shape(arr)[0] <= 20:  # ——————————————————————————————————————可能把正确的簇也过滤了
        #     continue
        # else:
        arr_median = Getmedianpoints(arr[:, :3])
        # 以中位点为种子点进行KDTREE搜索
        # 给点云加入种子点
        arr_getpoint = np.vstack((originaldf , arr_median))
        # 种子点位置
        seedpoint = np.shape(arr_getpoint)[0] - 1

        # 将矩阵用o3d显示出来
        pcd = o3d.geometry.PointCloud()  # 首先建立一个pcd类型的数据（这是open3d中的数据形式）
        pcd.points = o3d.utility.Vector3dVector(arr_getpoint)  # 将点云转换成open3d中的数据形式并用pcd来保存，以方便用open3d处理
        pcd.paint_uniform_color([0.5, 0.5, 0.5])  # 统一颜色
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)  # 建立kdtree索引
        # ——————混合搜索————————
        [k2, idx2, _] = pcd_tree.search_hybrid_vector_3d(pcd.points[seedpoint], 0.02, 200)
        np.asarray(pcd.colors)[idx2[:], :] = [0, 1, 0.8]  # 半径搜索结果并渲染为青色

        np_points = np.asarray(pcd.points)
        np_colors = np.asarray(pcd.colors)
        index_tree = np.where(np_colors[:, 2] == 0.8)[0]
        kdtree_arr = np_points[index_tree]

        if np.shape(kdtree_arr)[0] == 1 :
            continue
        # 对每个聚类进行特征向量判断
        mu = np.mean(kdtree_arr, 0)  # 计算均值
        np.cov((kdtree_arr - mu).T)
        evals, evcts = np.linalg.eig(np.cov((kdtree_arr - mu).T))
        evals, evcts = evals.real, evcts.real
        # 用特征向量长度
        ## 根据特征值，对其从大到小排序，获得序号
        sort_idx = np.argsort(evals)[::-1]  # argsort()从小到大排序
        evals_sort = evals[sort_idx]
        evcts_sort = evcts[sort_idx]
        # max
        eval_max = evals_sort[0]
        evct_max = evcts_sort[0]
        # middle
        eval_mid = evals_sort[1]
        evct_mid = evcts_sort[1]
        # min
        eval_min = evals_sort[2]
        evct_min = evcts_sort[2]

        max_evcts_len = np.linalg.norm(eval_max)
        min_evcts_len = np.linalg.norm(eval_min)
        mid_evcts_len = np.linalg.norm(eval_mid)

        x = max_evcts_len / max_evcts_len
        y = mid_evcts_len / max_evcts_len  # 管状特征都较小
        z = min_evcts_len / max_evcts_len

        # 管状特征得x1都趋近于0 ，y,z 趋近于0
        if y <= 0.5 and z <= 0.1:  # **********特征值进行阈值判断选出非管状关键点**********|||||||||||||||||||||||||||
            continue
        else:
            l += 1
            if l == 1:
                m = arr

            else:
                m = np.vstack((m, arr))

    return m

def main():
    work_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    file_pathname = '%s\\dataset\\Soybean3D'%work_dir#test
    excelpath = "%s\\dataset\\Simples_GT.xlsx"%work_dir # 要打开的excel文件
    indexname = '文件名'  # 要以哪一个名字为索引

    # 所有真值的角度，所有测量出来的角度
    #参数
    k_list = [0.2]#0.05,0.1,0.15,,0.25,0.3
    ra_list = [0.008]#0.002,0.004,0.006,,0.01,0.012,0.014
    D_list = [0.03]#0.015,0.02,0.025,,0.035,0.04

    dic_p, dic_r, dic_F, dic_R = {'R\P': ra_list}, {'R\P': ra_list}, {'R\P': ra_list}, {'D': D_list}
    for k in k_list : #提取前百分之几
        P_list, R_list, F_list = [], [], []
        for r in ra_list : #半径范围
            all_branchpoint = []
            cc = 0 #记载每个文件的序号
            for infile in os.listdir(file_pathname):
                cc += 1
                print(infile)
                print('K:%s_R:%s'%(k,r)+'_%s_Done!' % cc)
                # 1_正确样本构建
                Sample_arr,df_data = sampleReal_branchpoints_angle(file_pathname, infile, excelpath, indexname)
                #3_计算出关键点
                df = df_data
                # originaldf = df_data
                k_num = int(np.shape(df)[0] * k)
                key_point_df = keypoint_detact(df, k_num, r)  # 关键点提取
                # 聚类
                newarr1 = DBSCAN_label_without_Noise(key_point_df)
                newarr1 = PCA_delete_wrongpoints(df_data, newarr1)
                # 对关键点按Z进行从高到低排序
                if np.shape(newarr1 )[0] < 5:
                    TRUE_branch_ponit = np.zeros(np.shape(Sample_arr))
                    all_branchpoint.append(TRUE_branch_ponit)
                    TP_l,FP_l = 0,0
                    FN_l = np.shape(Sample_arr)[0]
                    if cc == 1:
                        TP = TP_l
                        FP = FP_l
                        FN = FN_l
                    else:
                        TP = TP_l + TP
                        FP = FP_l + FP
                        FN = FN_l + FN
                elif np.shape(newarr1)[0] >= 5 :
                    medianpoint_arr = []
                    for p in np.unique(newarr1[:, 3]):
                        # 计算出每个聚类的中心点
                        index = np.where(newarr1[:, 3] == p)
                        arr = newarr1[index][:, :3]
                        mp = Getmedianpoints(arr)
                        # mp = np.append(mp, p)
                        medianpoint_arr.append(mp)
                    medianpoint_arr = np.asarray(medianpoint_arr)
                    sort_idx = np.argsort(medianpoint_arr[:, 2])[::-1]  # argsort()从小到大排序
                    medianpoint_arr = medianpoint_arr[sort_idx]

                    TP_l,FP_l,FN_l,TRUE_branch_ponit = Keypoint_PRFestamate(medianpoint_arr, Sample_arr, threshold=0.01)
                    all_branchpoint.append(TRUE_branch_ponit)
                    if cc == 1:
                        TP = TP_l
                        FP = FP_l
                        FN = FN_l
                    else:
                        TP = TP_l + TP
                        FP = FP_l + FP
                        FN = FN_l + FN

            print('*****************(K：%s; R:%s)accuracy of branch detection：*****************' % (k, r))
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
                for infile in os.listdir(file_pathname):
                    count += 1
                    # 1_正确样本构建
                    Sample_arr, df_data = sampleReal_branchpoints_angle(file_pathname, infile, excelpath, indexname)
                    TRUE_branch_ponit = all_branchpoint[count]
                    # 3_通过正确的关键点计算角度
                    total_angle_vector = Branch_point_optimization(df_data, TRUE_branch_ponit, D=D)
                    # print(True_Predicted_arr[:,3])
                    # print(Sample_arr[:,3])
                    # 4_计算模型的关键节点F值以及角度R值
                    if count == 0:
                        all_real_angle = Sample_arr[:, 3]
                        all_angle_arr = total_angle_vector
                    else:
                        all_real_angle = np.append(all_real_angle, Sample_arr[:, 3])
                        all_angle_arr = np.append(all_angle_arr, total_angle_vector)
                index = np.where((all_angle_arr == 0))
                all_real_angle = np.delete(all_real_angle, index)
                all_angle_arr = np.delete(all_angle_arr, index)

                # 输出到excel中
                # dic = {'Measured angle': all_angle_arr,
                #        'Predicted angle': all_real_angle}
                # df = pd.DataFrame(dic)
                # df.to_csv('%s\\results\\DB_best_Fsocre.csv'%work_dir)

                R,RMSE = R2RMSE(all_real_angle, all_angle_arr)
                r_list.append(R)
                print('(D：%s)角度计算r值：' % D, R)

    # 输出到excel中,做参数敏感性分析
            dic_R['P:%s/R:%s' % (k, r)] = r_list
            df_R = pd.DataFrame(dic_R)  # pd.DataFrame.from_dict(df_p, orient=‘index’)
        dic_p['%s' % k], dic_r['%s' % k], dic_F['%s' % k] = P_list, R_list, F_list
    df_p, df_r, df_F = pd.DataFrame(dic_p), pd.DataFrame(dic_r), pd.DataFrame(dic_F)
    with pd.ExcelWriter('%s\\results\\DB_F_sensitivity.xlsx'%work_dir) as writer:
        df_p.to_excel(writer, sheet_name='Precision', index=False)
        df_r.to_excel(writer, sheet_name='Recall', index=False)
        df_F.to_excel(writer, sheet_name='F-score', index=False)
        df_R.to_excel(writer, sheet_name='r_value', index=False)



if __name__ == '__main__':
    main()