import xml.etree.ElementTree as ET

def SenticNetQuery(QString):
        root = ET.parse('processed2.xml').getroot()
        for descriptions in root:
            for elements in descriptions:
                if elements.tag=='text'and elements.text==QString:
                        return float(descriptions.find("polarity").text)
                        #print(float(descriptions.find("polarity").text))
                        #break
#print 'hi'
##                 
##       
##from lxml import etree
##print "enter queryword"
##Qword=raw_input()
##tree=etree.parse("senticnet3.rdf.xml")
####all_recipes=tree.xpath('./recipebook/recipe')
####recipe_names=[x.xpath('recipe_name/text()') for x in all_recipes]
####ingredients=[x.getparent().xpath('../ingredient_list/ingredients') for x in recipe_names]
####ingredient_names=[x.xpath('ingredient_name/text()') for x in ingredients]
##ns = {'RDF':'http://sentic.net/api'}
##for df in tree.xpath('//rdf:Description'):
##        ## subfield is a child of datafield, and iterate
##        subfields = df.getchildren()
##        for i in subfields (0, len(subfields)):
##            print i.tag
####        for subfield in subfields:
####                if subfield=='text' & subfield.text==Qword:
####                    subfields['polarity']
##
