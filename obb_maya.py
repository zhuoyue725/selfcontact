import numpy as np
import trimesh as tm

# 应用于2D、3D
def PCA(points):
    pointArray = np.array(points)
    ca = np.cov(pointArray,y = None,rowvar = 0,bias = 1)
    v, vect = np.linalg.eig(ca)
    tvect = np.transpose(vect)
    #use the inverse of the eigenvectors as a rotation matrix and
    #rotate the points so they align with the x and y axes
    ar = np.dot(pointArray,np.linalg.inv(tvect))
    
    # get the minimum and maximum x and y 
    mina = np.min(ar,axis=0)
    maxa = np.max(ar,axis=0)
    diff = (maxa - mina)*0.5
    # the center is just half way between the min and max xy
    center = mina + diff
    
    #get the 8 corners by subtracting and adding half the bounding boxes height and width to the center
    pointShape = pointArray.shape
    if pointShape[1] == 2:
        corners = np.array([center+[-diff[0],-diff[1]],
                        center+[diff[0],-diff[1]],
                        center+[diff[0],diff[1]],
                        center+[-diff[0],diff[1]],
                        center+[-diff[0],-diff[1]]])
    if pointShape[1] == 3:
        #get the 8 corners by subtracting and adding half the bounding boxes height and width to the center
        corners = np.array([center+[-diff[0],-diff[1],-diff[2]],
                    center+[diff[0],-diff[1],-diff[2]],                    
                    center+[diff[0],diff[1],-diff[2]],
                    center+[-diff[0],diff[1],-diff[2]],
                    center+[-diff[0],diff[1],diff[2]],
                    center+[diff[0],diff[1],diff[2]],                    
                    center+[diff[0],-diff[1],diff[2]],
                    center+[-diff[0],-diff[1],diff[2]],
                    center+[-diff[0],-diff[1],-diff[2]]])   
    
    #use the the eigenvectors as a rotation matrix and
    #rotate the corners and the centerback
    corners = np.dot(corners,tvect)
    center = np.dot(center,tvect)
    radius = diff
    if pointShape[1] == 2:
        array0,array1 = np.abs(vect[0,:]),np.abs(vect[1,:])
        index0,index1 = np.argmax(array0),np.argmax(array1)
        radius[index0],radius[index1] = diff[0],diff[1]
    if pointShape[1] == 3:
        array0,array1,array2 = np.abs(vect[0,:]),np.abs(vect[1,:]),np.abs(vect[2,:])
        index0,index1,index2 = np.argmax(array0),np.argmax(array1),np.argmax(array2)
        radius[index0],radius[index1],radius[index2] = diff[0],diff[1],diff[2]
    eigenvalue = v
    eigenvector = vect
    return corners, center, radius,eigenvalue,eigenvector


if "__main__" == __name__:
    obj = tm.load_mesh("/usr/pydata/t2m/selfcontact/output/extremities/bone_0.obj")
    obj_hull = obj.convex_hull
    corners, pcaCenter, pcaRadius,eigenvalue,eigenvector = PCA(obj.vertices)
    print("corners", corners)
    print("center", pcaCenter)
    print("radius", pcaRadius)
    print("eigenvalue", eigenvalue)
    print("eigenvector", eigenvector)
    
    # 创建长方体的面数组
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # 底部面
        [4, 5, 6], [4, 6, 7],  # 顶部面
        [0, 1, 5], [0, 5, 4],  # 前面
        [2, 3, 7], [2, 7, 6],  # 后面
        [0, 3, 7], [0, 7, 4],  # 左侧面
        [1, 2, 6], [1, 6, 5]   # 右侧面
    ])
    corners = corners[:8]
    # 创建box模型
    # bbox = tm.creation.box_corners(corners)
    combined_vertices = np.vstack([obj.vertices, corners])
    combined_faces = np.vstack([obj.faces, faces + len(obj.vertices)])
    combined_mesh = tm.Trimesh(vertices=combined_vertices, faces=combined_faces)
    # 将创建的box模型添加到原始模型中
    save_path = "/usr/pydata/t2m/selfcontact/output/tmp/obb_2.obj"
    # mesh = tm.util.concatenate(obj, bbox)
    combined_mesh.export(save_path)
