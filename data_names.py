import os

path_tr = "vectors/training/"
path_val = "vectors/validation/"

def has_all_files(prefix, num, path_list):
	return prefix + "-body-" + num in path_list and \
			prefix + "-doc-" + num in path_list and \
			prefix + "-occur-" + num in path_list and \
			prefix + "-labels-" + num in path_list


tr = os.listdir (path_tr)
val = os.listdir(path_val)

f = open ("vector_paths.txt", "w")

for filename in tr:
	path = path_tr
	visited = []
	prefix = "-".join(filename.split("-")[:-2])
	num = filename.split("-")[-1]
	if prefix+num not in visited:# and has_all_files (prefix, num, tr):
		visited.append(prefix+num)
		f.write(path + prefix + "-body-" + num + "," + path + prefix + "-doc-" + num +","+
			path + prefix + "-occur-" + num + "," + path + prefix + "-labels-" + num)
		f.write("\n")

for filename in val:
	path = path_val
	visited = []
	prefix = "-".join(filename.split("-")[:-2])
	num = filename.split("-")[-1]
	if prefix+num not in visited: #and has_all_files (prefix, num, tr):
		visited.append(prefix+num)
		f.write(path + prefix + "-body-" + num + "," + path + prefix + "-doc-" + num +","+
			path + prefix + "-occur-" + num + "," + path + prefix + "-labels-" + num)
		f.write("\n")


f.close()