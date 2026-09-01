nepse-auto-scanner

/ main.ymlname: NEPSE Scanner on: workflow dispatch:

schedule: - cron: "0 9 * * 1-5" jobs: scan: runs-on: ubuntu-latest

steps: - name: Checkout repository uses: actions

checkout@v4 - name: Setup Python uses: actions /
