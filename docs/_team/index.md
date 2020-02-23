---
title: The Team
---

# Team Members

{% for member in site.data.members %}
<div class="team-row"> 
<div class="team-row-image">
    <img src="{{site.baseurl}}/assets/team_photos/{{ member.photo }}" alt="{{ member.name }}">
</div>
<div class="team-row-bio">
    <div class="team-row-bio-links">
        <b> {{ member.name }} </b> <br>
        {{ member.institution }} <br>

        {% if member.website %} <a style="white-space: nowrap" href="website"> <i class="fa fa-home"></i> </a> {% endif %}
        {% if member.github %} <a style="white-space: nowrap" href="https://github.com/{{ member.github }}"> <i class="fa fa-github"></i> </a> {% endif %}
        {% if member.twitter %} <a style="white-space: nowrap" href="https://twitter.com/{{ member.twitter }}"> <i class="fa fa-twitter"></i>  </a> {% endif %}
        {% if member.email %} <a style="white-space: nowrap" href="mailto:{{ member.email }}"> <i class="fa fa-envelope"></i> </a> {% endif %}
        {% if member.linkedin %} <a style="white-space: nowrap" href="www.linkedin.com/in/{{ member.linkedin }}"> <i class="fa fa-linkedin"></i> </a> {% endif %}
    </div>
</div>

<br>
</div> 

<hr>
{% endfor %}