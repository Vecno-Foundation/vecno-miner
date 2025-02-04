echo REM When mining to a local node, you can drop the -s option. > ${1}/mine.bat
echo echo ============================================================ >> ${1}/mine.bat
echo echo = Running Vecno Miner with default .bat. Edit to configure = >> ${1}/mine.bat
echo echo ============================================================ >> ${1}/mine.bat
echo :start >> ${1}/mine.bat
echo ${1}.exe -a qrzs2hd6rtcx2zd4dzkzrpqjx4jg8ndmqqjle8j9cpp93gg059tludxxvvfqd -s n.seeder1.vecno.org >> ${1}/mine.bat
echo goto start >> ${1}/mine.bat


# target\release\vecno-miner -a vecno:qzj9kz0kmc3rxl9mw86mlda2cqmvp3xhavx9h2jud5ehdchvruql6ey64r8kz -s localhost
