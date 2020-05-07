---
title: News
permalink: /news/
sidebar: false
---

# News

<p>Subscribe with <a href="{{ site.baseurl }}/feed.xml">RSS</a> to keep up with the latest news about Turing.

{% for post in site.posts limit:10 %}
   {% if post.draft == null or post.draft == false %}
      <div class="post-preview">
      <h2><a href="{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a></h2>
      <span class="post-date">{{ post.date | date: "%B %d, %Y" }}</span><br>
      {% if post.badges %}{% for badge in post.badges %}<span class="badge badge-{{ badge.type }}">{{ badge.tag }}</span>{% endfor %}{% endif %}
      {{ post.excerpt }}
      <a href="{{ site.baseurl }}{{ post.url }}">read more</a>
      </div>
      <hr>
   {% endif %}
{% endfor %}

Want to see more? See the <a href="{{ site.baseurl }}/archive/">News Archive</a>.
