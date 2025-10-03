'''
Created on 8 Apr 2023

@author: Ragingwire
'''

import feedparser
NewsFeed = feedparser.parse("http://www.marketwatch.com/rss/topstories")
entry = NewsFeed.entries[1]

print (entry.keys())

NumPosts = len (NewsFeed.entries )
print ('Number of RSS posts :', NumPosts )

for post in range ( 0, NumPosts ):
    entry = NewsFeed.entries [ post ]
    
    print ('Post Title :',entry.title)

    print (entry.published)
    print ("******")
    print (entry.summary)
    # print ("------News Link--------")
    # print (entry.link)