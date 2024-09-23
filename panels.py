# ### PANELS
# 

# In[ ]:


lists = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
gfed_cover_labels = {
        0: "BONA",
        1: "TOTAL",
        2: "BOAS",
        3: "TENA",
        4: "EURO",
        5: "CEAS",
        6: "CEAM",
        7: "MIDE",
        8: "SEAS",
        9: "NHSA",
        10: "NHAF",
        11: "EQAS",
        12: "SHSA",
        13: "SHAF",
        14: "AUST"
    }

img_paths =[]
for i in lists:
    #add path to the files you want in the panel and appaend them to the list
    #img_paths.append(f'C:/Users/ssenthil/Desktop/Datasets/Model_plots/MODEL_combined/BA_SEASON_combied_{gfed_cover_labels[i]}_1997_2020.png')


# In[ ]:


import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

# Ensure the output directory exists

#os.makedirs(output_dir, exist_ok=True)
#os.chdir(output_dir)
#specify output directory


# Define the number of plots and rows
total_plots = 15  #give own val
rows = 5 #give own val
cols = 3 #give own val

fig, axs = plt.subplots(rows, cols, figsize=(20, 15))
for i in range(rows):
    for j in range(cols):
        index = i * cols + j
        
        if index >= total_plots:
            break  # Stop if all plots are processed
        
        img_path = img_paths[index]
        img = mpimg.imread(img_path)
        axs[i, j].imshow(img)
        axs[i, j].axis('off')
        # Add any additional customizations to the subplots if needed

# Hide any empty subplots
for i in range(total_plots, rows * cols):
    axs.flatten()[i].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'Model_Combined_Season.png'), dpi=600)  # specify fle name
plt.close()