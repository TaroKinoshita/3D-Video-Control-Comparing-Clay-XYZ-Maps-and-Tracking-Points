# 結合方法（PowerShell は cmd /c copy /b、bash は cat）
cat rq2_results.tar.gz.part_a? > rq2_results.tar.gz
md5sum -c rq2_results.tar.gz.md5
