# -*- coding: utf-8 -*-
"""
---------------------------------------
@Time    : 2024-06-22 13:10
@Author  : lijing
@File    : schedule.py
@Description: 定时任务
---------------------------------------
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

pool = ThreadPoolExecutor()


class Schedule:
    """
    目前为单个任务，待扩展
    """

    def __init__(self, task_logic, interval=60) -> None:
        self.task = task_logic
        self.stopped = False
        self.interval = interval

    def run_task(self):
        self.task()

    async def loop(self):
        while not self.stopped:
            self.run_task()
            await asyncio.sleep(self.interval)

    def start_loop(self):
        asyncio.run(self.loop())

    def start_loop_async(self):
        pool.submit(self.start_loop)

    def stop_loop(self):
        self.stopped = True
