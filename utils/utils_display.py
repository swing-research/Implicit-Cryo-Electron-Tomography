import numpy as np
import torch
import matplotlib.pyplot as plt


"""
field: implicit representation that given a set of coordinates return the displacement in these points.
field_true: optional, if not None, this field in added on the same plot as the true field.
Npts: number of points where to observe deformation in each dimension.
img_path: path where to save the image (without extension)
img_type: format of the image, pdf or png.
scale, alpha, width: parameters of the plot, see quiver documentation.
"""
def display_local(field,field_true=None,Npts=(10,10),img_path='',img_type='.pdf',scale=3,alpha=0.8,width=0.002,device='cuda',wx=1,wy=1):
    ## Display quiver
    xx1 = torch.linspace(-1,1,Npts[0],device=device)
    xx2 = torch.linspace(-1,1,Npts[1],device=device)
    XX_t, YY_t = torch.meshgrid(xx1,xx2,indexing='ij')
    XX_t = torch.unsqueeze(XX_t, dim = 2)
    YY_t = torch.unsqueeze(YY_t, dim = 2)
    coordinates = torch.cat([XX_t,YY_t],2).reshape(-1,2)
    displacement = field(coordinates)
    x2 = coordinates[:,0].detach().cpu().numpy()*wx
    y2 = coordinates[:,1].detach().cpu().numpy()*wy
    u2 = displacement[:,0].detach().cpu().numpy()
    v2 = displacement[:,1].detach().cpu().numpy()
    plt.figure(1)
    plt.clf()
    plt.tight_layout()
    ax = plt.gca()
    ax.quiver(x2,y2,u2,v2,angles='xy',scale=scale,alpha=alpha,color='r',width=width,label='Estimation')
    # u1 = field.depl_ctr_pts[0,0].detach().cpu().numpy().reshape(-1)
    # v1 = field.depl_ctr_pts[0,1].detach().cpu().numpy().reshape(-1)
    # x1 = field.coordinates[0].detach().cpu().numpy().reshape(-1)
    # y1 = field.coordinates[1].detach().cpu().numpy().reshape(-1)
    # ax.quiver(x1,y1,u1,v1,angles='xy',scale=scale,alpha=alpha,color='k',linestyle=':',width=width,label='Estimation ref')
    if img_path!='' and field_true is None:
        plt.savefig(img_path+img_type)
    if img_path!='' and field_true is not None:
        plt.savefig(img_path+'_est'+img_type)
    # plt.axis('off')
    if field_true is not None:
        displacement = field_true(coordinates)
        x2 = coordinates[:,0].detach().cpu().numpy()*wx
        y2 = coordinates[:,1].detach().cpu().numpy()*wy
        u2 = displacement[:,0].detach().cpu().numpy()
        v2 = displacement[:,1].detach().cpu().numpy()
        ax = plt.gca()
        ax.quiver(x2,y2,u2,v2,angles='xy',scale=scale,alpha=alpha,color='b',width=width,label='True')
        # u1 = field_true.depl_ctr_pts[0,0].detach().cpu().numpy().reshape(-1)
        # v1 = field_true.depl_ctr_pts[0,1].detach().cpu().numpy().reshape(-1)
        # x1 = field_true.coordinates[0].detach().cpu().numpy().reshape(-1)
        # y1 = field_true.coordinates[1].detach().cpu().numpy().reshape(-1)
        # ax.quiver(x1,y1,u1,v1,angles='xy',scale=scale,alpha=alpha,color='k',linestyle=':',width=width,label='True ref')
        plt.legend()
        if img_path!='' and field_true is not None:
            plt.savefig(img_path+'_est_and_true'+img_type)
        if img_path!='':
            plt.figure(1)
            plt.clf()
            plt.tight_layout()
            ax = plt.gca()
            ax.quiver(x2,y2,u2,v2,angles='xy',scale=scale,alpha=alpha,color='b',width=width,label='True')
            # u1 = field_true.depl_ctr_pts[0,0].detach().cpu().numpy().reshape(-1)
            # v1 = field_true.depl_ctr_pts[0,1].detach().cpu().numpy().reshape(-1)
            # x1 = field_true.coordinates[0].detach().cpu().numpy().reshape(-1)
            # y1 = field_true.coordinates[1].detach().cpu().numpy().reshape(-1)
            # ax.quiver(x1,y1,u1,v1,angles='xy',scale=scale,alpha=alpha,color='k',linestyle=':',width=width,label='True ref')
            if img_path!='' and field_true is not None:
                plt.savefig(img_path+'_true'+img_type)


def display_local_movie(field,field_true=None,Npts=(10,10),img_path='',img_type='.pdf',scale=3,alpha=0.8,width=0.002,
                        device='cuda',loc='upper right',legend1='Estimation', legend2='True'):
    xx1 = torch.linspace(-1,1,Npts[0],device=device)
    xx2 = torch.linspace(-1,1,Npts[1],device=device)
    XX_t, YY_t = torch.meshgrid(xx1,xx2,indexing='ij')
    XX_t = torch.unsqueeze(XX_t, dim = 2)
    YY_t = torch.unsqueeze(YY_t, dim = 2)
    coordinates = torch.cat([XX_t,YY_t],2).reshape(-1,2)
    for k in range(len(field)):
        ## Display quiver
        displacement = field[k](coordinates)
        x2 = coordinates[:,0].detach().cpu().numpy()
        y2 = coordinates[:,1].detach().cpu().numpy()
        u2 = displacement[:,0].detach().cpu().numpy()
        v2 = displacement[:,1].detach().cpu().numpy()
        plt.figure(1)
        plt.clf()
        ax = plt.gca()
        ax.quiver(x2,y2,u2,v2,angles='xy',scale=scale,alpha=alpha,color='r',width=width,label=legend1)
        plt.xticks(np.linspace(-1,1,5), [-1,-0.5,0,0.5,1])
        plt.yticks(np.linspace(-1,1,5), [-1,-0.5,0,0.5,1])
        plt.tight_layout()
        # u1 = field[k].depl_ctr_pts[0,0].detach().cpu().numpy().reshape(-1)
        # v1 = field[k].depl_ctr_pts[0,1].detach().cpu().numpy().reshape(-1)
        # x1 = field[k].coordinates[0].detach().cpu().numpy().reshape(-1)
        # y1 = field[k].coordinates[1].detach().cpu().numpy().reshape(-1)
        # ax.quiver(x1,y1,u1,v1,angles='xy',scale=scale,alpha=alpha,color='k',linestyle=':',width=width,label='Estimation ref')
        if img_path!='' and field_true is None:
            plt.savefig(img_path+str(k)+img_type)
        if img_path!='' and field_true is not None:
            plt.savefig(img_path+'est'+str(k)+img_type)
        # plt.axis('off')
        if field_true is not None:
            displacement = field_true[k](coordinates)
            x2 = coordinates[:,0].detach().cpu().numpy()
            y2 = coordinates[:,1].detach().cpu().numpy()
            u2 = displacement[:,0].detach().cpu().numpy()
            v2 = displacement[:,1].detach().cpu().numpy()
            ax = plt.gca()
            ax.quiver(x2,y2,u2,v2,angles='xy',scale=scale,alpha=alpha,color='b',width=width,label=legend2)
            # u1 = field_true[k].depl_ctr_pts[0,0].detach().cpu().numpy().reshape(-1)
            # v1 = field_true[k].depl_ctr_pts[0,1].detach().cpu().numpy().reshape(-1)
            # x1 = field_true[k].coordinates[0].detach().cpu().numpy().reshape(-1)
            # y1 = field_true[k].coordinates[1].detach().cpu().numpy().reshape(-1)
            # ax.quiver(x1,y1,u1,v1,angles='xy',scale=scale,alpha=alpha,color='k',linestyle=':',width=width,label='True ref')
            plt.legend(loc=loc)
            plt.xticks(np.linspace(-1,1,5), [-1,-0.5,0,0.5,1])
            plt.yticks(np.linspace(-1,1,5), [-1,-0.5,0,0.5,1])
            plt.tight_layout()
            if img_path!='' and field_true is not None:
                plt.savefig(img_path+'est_and_true'+str(k)+img_type)
            if img_path!='':
                plt.figure(1)
                plt.clf()
                ax = plt.gca()
                ax.quiver(x2,y2,u2,v2,angles='xy',scale=scale,alpha=alpha,color='b',width=width,label=legend2)
                # u1 = field_true[k].depl_ctr_pts[0,0].detach().cpu().numpy().reshape(-1)
                # v1 = field_true[k].depl_ctr_pts[0,1].detach().cpu().numpy().reshape(-1)
                # x1 = field_true[k].coordinates[0].detach().cpu().numpy().reshape(-1)
                # y1 = field_true[k].coordinates[1].detach().cpu().numpy().reshape(-1)
                # ax.quiver(x1,y1,u1,v1,angles='xy',scale=scale,alpha=alpha,color='k',linestyle=':',width=width,label='True ref')
                plt.xticks(np.linspace(-1,1,5), [-1,-0.5,0,0.5,1])
                plt.yticks(np.linspace(-1,1,5), [-1,-0.5,0,0.5,1])
                plt.tight_layout()
                if img_path!='' and field_true is not None:
                    plt.savefig(img_path+'true'+str(k)+img_type)

