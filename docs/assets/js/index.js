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

    if (typeof(window.DOC_VERSIONS) == "undefined") {
        window.DOC_VERSIONS = ["dev"];
    }
    var url_parts = /(.*:\/\/[^/]+\/)(.+?)(\/.*)/.exec(window.location.href);

    $(".dropdown > a").text(url_parts[2]);
    $.each(DOC_VERSIONS, function(index, value) {
        if(value == url_parts[2]) {
            // mobile
            $("select#version-selector").append(
                $('<option value="' + value + '" selected="selected">' + value + '</option>'));
            return;
        }
        // desktop
        $(".dropdown > div.dropdown-menu").append(
            $('<a class="dropdown-item" href="#">' + value + '</a>'));
        // mobile
        $("select#version-selector").append(
            $('<option value="' + value + '">' + value + '</option>'));
    });

    $(".dropdown > div.dropdown-menu > a").on("click", function() {
        var target_version = $(this).text().trim();
        if (target_version == url_parts[2]) return;
        window.location.href = url_parts[1] + target_version + url_parts[3];
    });

    $("select#version-selector").change(function() {
        var target_version = $("select#version-selector option:selected").text();
        if (target_version == url_parts[2]) return;
        window.location.href = url_parts[1] + target_version + url_parts[3];
    });
});
