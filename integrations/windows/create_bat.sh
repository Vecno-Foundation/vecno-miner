echo REM When mining to a local node, you can drop the -s option. > ${1}/mine.bat
echo echo ============================================================ >> ${1}/mine.bat
echo echo = Running Vecno Miner with default .bat. Edit to configure = >> ${1}/mine.bat
echo echo ============================================================ >> ${1}/mine.bat
echo :start >> ${1}/mine.bat
echo ${1}.exe -a qrzs2hd6rtcx2zd4dzkzrpqjx4jg8ndmqqjle8j9cpp93gg059tludxxvvfqd -s n.seeder1.vecno.org >> ${1}/mine.bat
echo goto start >> ${1}/mine.bat


# target\release\vecno-miner -a vecno:qqtsqwxa3q4aw968753rya4tazahmr7jyn5zu7vkncqlvk2aqlsdsah9ut65e -s localhost
