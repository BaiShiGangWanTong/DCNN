import os


cwd = r'B:\\PycharmProjects\DCNN\\train\\fire'
res = os.listdir(cwd)
print(res[0])
print("----------------------------")

for i in range(len(res)):
    os.rename(os.path.join(cwd, res[i]), os.path.join(cwd, "fire" + str(i).zfill(6) + '.png'))

print("done!")