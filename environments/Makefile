default.conda-lock.yml: default.yml
	conda-lock -p linux-64 -p win-64 -f default.yml --lockfile default.conda-lock.yml

minimal.conda-lock.yml: minimal.yml
	conda-lock -p linux-64 -f minimal.yml --lockfile minimal.conda-lock.yml

minimal-linux-64.lock: minimal.conda-lock.yml
	conda-lock render minimal.conda-lock.yml -p linux-64 && mv conda-linux-64.lock minimal-linux-64.lock
