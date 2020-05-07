module Jekyll
  module Tags
    class WrapTag < Liquid::Block

      def initialize(tag_name, markup, tokens)
        super
        @class = markup
      end

      def render(context)
        site = context.registers[:site]
        converter = site.find_converter_instance(::Jekyll::Converters::Markdown)
        # below Jekyll 3.x use this:
        # converter = site.getConverterImpl(::Jekyll::Converters::Markdown)
        body = converter.convert(super(context))
        "<div class=\"#{@class}\">#{body}#{}</div>"
      end

    end
  end
end

Liquid::Template.register_tag('wrap', Jekyll::Tags::WrapTag)
