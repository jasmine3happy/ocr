import os
import zipfile
yadir='./submit'
zipfilepath='./submit.zip'
filelists = os.listdir(yadir)
if '.DS_Store' in filelists:
    filelists.remove('.DS_Store')
if filelists == None or len(filelists) < 1:
    print (">>>>>>待压缩的文件目录：" + yadir + " 里面不存在文件,无需压缩. <<<<<<")
else:
    z = zipfile.ZipFile(zipfilepath, 'w', zipfile.ZIP_DEFLATED)
    for fil in filelists:
        filefullpath = os.path.join(yadir, fil)
        # filefullpath是文件的全路径，fil是文件名，这样就不会带目录啦
        z.write(filefullpath, fil)
    z.close()