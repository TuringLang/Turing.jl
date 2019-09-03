---
layout: null
excluded_in_search: true
---
$(document).ready(function(){
    $('.md-header').attr('data-md-state', 'none');

    $(window).scroll(function() {
        if ($(window).scrollTop() > 20) {
            $('.md-header').attr('data-md-state', 'shadow');
        } else {
            $('.md-header').attr('data-md-state', 'none');
        }
    });
});
